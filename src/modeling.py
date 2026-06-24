from __future__ import annotations

import copy

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from omegaconf import DictConfig
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader
from torchvision.models import resnet50

from dataset import DecodeJPEGBytesDataset

IMAGENET1K_MEAN = (0.485, 0.456, 0.406)
IMAGENET1K_STD = (0.229, 0.224, 0.225)


class GatedAttention(nn.Module):
    def __init__(self, hidden_dim: int, attn_dim: int, dropout_rate: float = 0.25):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, attn_dim)
        self.w2 = nn.Linear(hidden_dim, attn_dim)
        self.w3 = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, use_dropout: bool = True) -> torch.Tensor:
        x1, x2 = self.w1(x).sigmoid(), self.w2(x).tanh()
        if self.training and use_dropout:
            x1, x2 = self.dropout(x1), self.dropout(x2)
        return self.w3(x1 * x2).squeeze(-1)


class VisionEncoder(nn.Module):
    """ImageNet-pretrained ResNet50 backbone."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*list(resnet50(pretrained=True).children())[:-3])
        self.register_buffer("mean", torch.tensor(IMAGENET1K_MEAN)[None, :, None, None])
        self.register_buffer("std", torch.tensor(IMAGENET1K_STD)[None, :, None, None])

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return (x.cuda().float().permute(0, 3, 1, 2) / 0xFF - self.mean) / self.std

    def forward(
        self,
        bytes_list: list[bytes],
        batch_size: int | None = None,
        num_workers: int = 8,
    ) -> torch.Tensor:
        self.layers.eval()

        dataset = DecodeJPEGBytesDataset(bytes_list)
        if batch_size is None:
            img = self._preprocess(torch.stack(list(tqdm.tqdm(dataset, leave=False))))
            return self.layers(img).mean((-2, -1))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
        )
        all_features = []
        for img in tqdm.tqdm(dataloader, leave=False):
            img = self._preprocess(img)
            with torch.autocast("cuda", torch.float16):
                features = self.layers(img)
            all_features.append(features.float().mean((-2, -1)))
        return torch.cat(all_features)


class VisionEncoder_UNI(nn.Module):
    """UNI foundation-model backbone (https://huggingface.co/MahmoodLab/UNI).

    Requires access to the gated HF repo. Set the ``HF_TOKEN`` environment variable
    (or run ``huggingface-cli login``) before instantiating this encoder.
    """

    def __init__(self):
        super().__init__()
        self.layers = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        self.layers.set_grad_checkpointing(True)
        self.transform = create_transform(
            **resolve_data_config(self.layers.pretrained_cfg, model=self.layers)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x.cuda().float().permute(0, 3, 1, 2) / 0xFF)

    def forward(
        self,
        bytes_list: list[bytes],
        batch_size: int | None = None,
        num_workers: int = 8,
    ) -> torch.Tensor:
        self.layers.eval()

        dataset = DecodeJPEGBytesDataset(bytes_list)
        if batch_size is None:
            img = self._preprocess(torch.stack(list(tqdm.tqdm(dataset, leave=False))))
            return self.layers(img)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=1,
        )
        all_features = []
        for img in tqdm.tqdm(dataloader, leave=False):
            img = self._preprocess(img)
            with torch.autocast("cuda", torch.float16):
                features = self.layers(img)
            all_features.append(features.float())
        return torch.cat(all_features)


class MMPL(nn.Module):

    def __init__(self, config: DictConfig):
        super().__init__()
        self.temperature = config.model.temperature
        self.sinkhorn_weight = config.model.sinkhorn_weight
        self.retrieve_k = config.model.retrieve_k
        self.encoder_batch_size = config.model.encoder_batch_size
        self.encoder_type = config.model.backbone

        if self.encoder_type == "ResNet50":
            self.encoder = VisionEncoder()
        elif self.encoder_type == "UNI":
            self.encoder = VisionEncoder_UNI()

        if hasattr(self, "encoder"):
            self.encoder_ema = copy.deepcopy(self.encoder).requires_grad_(False)

        self.ga = GatedAttention(
            hidden_dim=config.model.hidden_dim,
            attn_dim=config.model.attn_dim,
            dropout_rate=config.model.attn_dropout_rate,
        )
        self.w1 = nn.Linear(config.model.input_dim, config.model.hidden_dim)
        self.w2 = nn.Linear(config.model.hidden_dim, config.model.num_classes)

        self.in_dropout = nn.Dropout(config.model.input_dropout_rate)
        self.ret_dropout = nn.Dropout(config.model.retrieval_dropout_rate)

        self.prototypes = nn.Parameter(
            torch.randn(config.model.num_prototypes, config.model.input_dim)
        )
        self.register_buffer(
            "queue",
            torch.randn(config.model.queue_size, config.model.input_dim),
            persistent=False,
        )
        self.max_queue_update = int(
            config.model.queue_size * config.model.max_queue_update_ratio
        )
        self._queue_initialized = 0

    @torch.no_grad()
    def update_ema(self, ema_decay: float):
        src, dst = self.encoder.state_dict(), self.encoder_ema.state_dict()
        state_dict = {n: (1 - ema_decay) * src[n] + ema_decay * dst[n] for n in src}
        self.encoder_ema.load_state_dict(state_dict)

    def forward(
        self, inputs: torch.Tensor | list[bytes], labels: torch.Tensor
    ) -> tuple[dict, torch.Tensor, torch.Tensor, list[int]]:
        if self.encoder_type == "None":
            assert not isinstance(inputs, list)
            features = inputs
        else:
            features = (
                self.encoder_ema(inputs, self.encoder_batch_size)
                if isinstance(inputs, list)
                else inputs
            )

        if self.training:
            idx = torch.randperm(features.size(0), device=features.device)
            features = features[idx]
            if isinstance(inputs, list):
                inputs = [inputs[i] for i in idx.tolist()]
            if self._queue_initialized < self.queue.size(0):
                new_queue = min(self.queue.size(0), features.size(0))
                self._queue_initialized += new_queue
            else:
                new_queue = min(self.max_queue_update, features.size(0))
            self.queue.copy_(self.queue.roll(new_queue, 0))
            self.queue[:new_queue] = features[:new_queue].detach()

        prototypes_norm = F.normalize(self.w1(self.prototypes), dim=-1)
        features_norm = F.normalize(self.w1(features), dim=-1)
        all_features_norm = self.queue[: self._queue_initialized]
        all_features_norm = F.normalize(self.w1(all_features_norm), dim=-1)

        sims = torch.einsum("bd,kd->bk", features_norm, prototypes_norm)
        loss_sinkhorn = sims.new_zeros(())
        if self.training:
            sims_all = torch.einsum("bd,kd->bk", all_features_norm, prototypes_norm)
            pseudo_labels = sinkhorn(sims_all)[:new_queue]
            loss_sinkhorn = F.cross_entropy(
                sims[:new_queue] / self.temperature, pseudo_labels
            )

        coeffs = self.ga(prototypes_norm, use_dropout=False)
        coeffs = (coeffs * coeffs.size(0) / self.retrieve_k).softmax(0)
        num_samples = distribute_integers(coeffs, self.retrieve_k).tolist()

        sims = self.ret_dropout(sims + 1)

        candidates = []
        if features.size(0) > self.retrieve_k:
            for sims, k in zip(sims.T, num_samples):
                candidates.extend(sims.topk(k=k, dim=0).indices.tolist())
        else:
            print(f"[*] Too small! -- ({features.size(0)}, {self.retrieve_k})")
            candidates = list(range(features.size(0)))

        if self.encoder_type == "None":
            features = features[candidates]
        else:
            features = (
                self.encoder([inputs[i] for i in candidates])
                if isinstance(inputs, list)
                else features[candidates]
            )

        features = self.w1(features)
        features = self.in_dropout(features)
        logits = self.w2(self.ga(F.normalize(features, dim=-1)).softmax(0) @ features)
        loss_ce = F.cross_entropy(logits, labels)

        if self.training:
            metrics = {
                "loss_ce": loss_ce,
                "sinkhorn_loss": loss_sinkhorn,
                "loss": loss_ce + self.sinkhorn_weight * loss_sinkhorn,
            }
        else:
            metrics = {"loss": loss_ce, "loss_ce": loss_ce}
        return metrics, logits.softmax(-1), logits.argmax(-1), num_samples


def distribute_integers(probs: torch.Tensor, total: int) -> torch.Tensor:
    """Split `total` into per-prototype integer counts proportional to `probs`."""
    amounts = (probs * total).floor().long()
    if (remainder := total - amounts.sum().item()) > 0:
        amounts[(probs * total - amounts).sort(descending=True)[1][:remainder]] += 1
    return amounts


@torch.no_grad()
def sinkhorn(
    logits: torch.Tensor, epsilon: float = 0.05, iterations: int = 3
) -> torch.Tensor:
    """Sinkhorn-Knopp normalization producing a uniform assignment over prototypes."""
    Q = torch.exp(logits / epsilon).t()
    Q /= Q.sum()
    for _ in range(iterations):
        Q /= Q.sum(dim=1, keepdim=True) * Q.size(0)
        Q /= Q.sum(dim=0, keepdim=True) * Q.size(1)
    return (Q * Q.size(1)).t()
