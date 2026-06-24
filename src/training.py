from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import torch
import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

import modeling
from dataset import PatchFeatureDataset
from utils import MetricLogger, ModelSaver, TensorBoardLogger


class Trainer:
    def __init__(self, config: DictConfig, datasets: tuple[PatchFeatureDataset, ...]):
        self.config = config
        self.max_epochs = config.experiment.max_epochs
        self.batch_size = config.data.batch_size
        self.ema_decay = config.model.ema_decay
        self.reduce_after = config.optim.reduce_after
        self.model = getattr(modeling, config.model.name)(config)

        # Checkpoint paths.
        self.checkpoint_path = os.path.join(
            config.experiment.directory, "model_best_loss.pt"
        )
        self.last_checkpoint_path = os.path.join(
            config.experiment.directory, "model_last.pt"
        )

        # Initialize Weights & Biases logging if enabled.
        if hasattr(config, "wandb") and config.wandb.use:
            wandb.init(
                project=config.wandb.project,
                name=config.wandb.name
                or f"{os.path.basename(config.experiment.directory)}",
                config=OmegaConf.to_container(config, resolve=True),
                dir=config.wandb.dir or os.getcwd(),
            )

        self.logger = TensorBoardLogger(config.experiment.directory)
        self.saver = ModelSaver(
            directory=config.experiment.directory,
            metric=config.experiment.save_metric,
            mode=config.experiment.save_mode,
        )
        self._initialize_dataloaders(*datasets)
        self._configure_optimizer()

        # Resume training when requested or when a checkpoint already exists.
        self.start_epoch = 0
        if config.experiment.resume or (
            os.path.isfile(self.checkpoint_path)
            or os.path.isfile(self.last_checkpoint_path)
        ):
            self._load_checkpoint()

    def _initialize_dataloaders(
        self,
        train_dataset: PatchFeatureDataset,
        valid_dataset: PatchFeatureDataset,
        test_dataset: PatchFeatureDataset,
    ):
        train_args = dict(batch_size=1, num_workers=2, prefetch_factor=1, shuffle=True)
        valid_args = dict(batch_size=1, num_workers=2, prefetch_factor=1)

        if self.config.data.use_weighted_sampler:
            weights_per_class = {
                k: len(train_dataset) / v
                for k, v in Counter(train_dataset.labels).items()
            }
            weights = [weights_per_class[i] for i in train_dataset.labels]
            train_args["sampler"] = WeightedRandomSampler(weights, len(weights))
            train_args["shuffle"] = False

        self.train_dataloader = DataLoader(train_dataset, **train_args)
        self.valid_dataloader = DataLoader(valid_dataset, **valid_args)
        self.test_dataloader = DataLoader(test_dataset, **valid_args)

    def _configure_optimizer(self):
        mil_params, encoder_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "encoder.layers" in n:
                encoder_params.append(p)
            else:
                mil_params.append(p)

        param_groups = [
            {"params": mil_params, "lr": self.config.optim.mil_lr},
            {"params": encoder_params, "lr": self.config.optim.encoder_lr},
        ]
        self.optimizer = Adam(param_groups, weight_decay=self.config.optim.weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.optim.reduce_factor,
            patience=self.config.optim.reduce_patience,
        )

    def _load_checkpoint(self):
        """Load model, optimizer, scheduler, and epoch state from a checkpoint."""
        checkpoint_file = (
            self.last_checkpoint_path
            if os.path.isfile(self.last_checkpoint_path)
            else self.checkpoint_path
        )

        if not os.path.isfile(checkpoint_file):
            print(f"=> No checkpoint found at: {checkpoint_file}")
            return

        print(f"=> Loading checkpoint: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)

        # Load the model weights.
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("=> Loaded model weights")
        else:
            # Support the legacy checkpoint format (state dict only).
            self.model.load_state_dict(checkpoint)
            print("=> Loaded model weights (legacy format)")
            return

        # Load the optimizer state.
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("=> Loaded optimizer state")
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        # Load the scheduler state.
        if "scheduler_state_dict" in checkpoint and hasattr(self, "scheduler"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("=> Loaded learning-rate scheduler state")

        # Resume from the next epoch.
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1
            print(f"=> Resuming from epoch {self.start_epoch}")

        # Load the EMA encoder state when available.
        if (
            "ema_state_dict" in checkpoint
            and hasattr(self.model, "encoder")
            and hasattr(self.model.encoder, "encoders_ema")
        ):
            for key, ema_encoder in self.model.encoder.encoders_ema.items():
                if key in checkpoint["ema_state_dict"]:
                    ema_encoder.load_state_dict(checkpoint["ema_state_dict"][key])
            print("=> Loaded EMA model state")

        print(f"=> Checkpoint loaded (epoch {self.start_epoch - 1})")

    def _save_checkpoint(self, epoch):
        """Save the current training state as the latest checkpoint."""
        ema_state_dict = {}
        if hasattr(self.model, "encoder") and hasattr(
            self.model.encoder, "encoders_ema"
        ):
            for key, ema_encoder in self.model.encoder.encoders_ema.items():
                ema_state_dict[key] = ema_encoder.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if hasattr(self, "scheduler") else None
            ),
            "ema_state_dict": ema_state_dict,
        }
        torch.save(checkpoint, self.last_checkpoint_path)

    def _train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        total_metrics, loss_accum = defaultdict(list), []

        for i, (inputs, labels) in tqdm.tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc="Train",
        ):
            inputs = inputs[0].cuda() if isinstance(inputs, torch.Tensor) else inputs
            metrics, _, _, _ = self.model(inputs, labels[0].cuda())

            for k, v in metrics.items():
                total_metrics[k].append(v.item())
            loss_accum.append(metrics["loss"])

            # Every `self.batch_size` steps, update the model by averaging the previous
            # `self.batch_size` step losses. This is a form of gradient accumulation,
            # since the number of patches differs from slide to slide.
            if (i + 1) % self.batch_size == 0 or i == len(self.train_dataloader) - 1:
                (sum(loss_accum) / len(loss_accum)).backward()
                loss_accum = []
                self.optimizer.step()
                self.optimizer.zero_grad()

                if 0 < self.ema_decay < 1:
                    self.model.update_ema(self.ema_decay)

        return {f"train/{k}": sum(v) / len(v) for k, v in total_metrics.items()}

    @torch.no_grad()
    def _valid_one_epoch(self):
        self.model.eval()
        total_metrics, valid_logger = defaultdict(list), MetricLogger()

        for i, (inputs, labels) in tqdm.tqdm(
            enumerate(self.valid_dataloader),
            total=len(self.valid_dataloader),
            desc="Valid",
        ):
            inputs = inputs[0].cuda() if isinstance(inputs, torch.Tensor) else inputs
            metrics, probs, preds, num_samples = self.model(inputs, labels[0].cuda())
            for k, v in metrics.items():
                total_metrics[k].append(v.item())
            valid_logger.log(probs.cpu(), preds.cpu(), labels[0])

        summary = {f"val/{k}": sum(v) / len(v) for k, v in total_metrics.items()}
        summary |= {f"val/{k}": v for k, v in valid_logger.summary(True).items()}
        print(f"* Prototype Retrieve K: {num_samples}")
        self.saver(self.model, summary)
        return summary

    def train(self):
        self.model.cuda()
        for epoch in tqdm.trange(self.start_epoch, self.max_epochs, desc="Epochs"):
            try:
                train_summary = self._train_one_epoch(epoch)
                valid_summary = self._valid_one_epoch()

                if epoch >= self.reduce_after:
                    self.scheduler.step(valid_summary["val/loss"])
                for i, pg in enumerate(self.optimizer.param_groups):
                    train_summary[f"train/learning_rate/pg{i}"] = pg["lr"]

                self.logger("log_scalars", summary=train_summary, step=epoch)
                self.logger("log_scalars", summary=valid_summary, step=epoch)

                self._save_checkpoint(epoch)

                if wandb.run is not None:
                    wandb_metrics = {**train_summary, **valid_summary, "epoch": epoch}
                    wandb.log(wandb_metrics)

            except KeyboardInterrupt:
                break

        if wandb.run is not None:
            wandb.finish()

    @torch.no_grad()
    def test(self, use_best_model=True):
        """Evaluate on the test set.

        Args:
            use_best_model: load the best checkpoint if True, otherwise the last one.
        """
        checkpoint_path = (
            self.checkpoint_path if use_best_model else self.last_checkpoint_path
        )

        if os.path.isfile(checkpoint_path):
            print(f"=> Loading model for testing: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            print(f"Warning: no checkpoint found for testing: {checkpoint_path}")

        self.model.cuda()
        self.model.eval()

        test_logger = MetricLogger()
        for i, (inputs, labels) in tqdm.tqdm(
            enumerate(self.test_dataloader), desc="Test"
        ):
            inputs = inputs[0].cuda() if isinstance(inputs, torch.Tensor) else inputs
            _, probs, preds, _ = self.model(inputs, labels[0].cuda())
            test_logger.log(probs.cpu(), preds.cpu(), labels.cpu())

        print("*** TEST ***")
        test_summary = {f"test/{k}": v for k, v in test_logger.summary(True).items()}
        self.logger("log_scalars", summary=test_summary, step=0)

        if wandb.run is not None:
            wandb.log(test_summary)

        output_path = checkpoint_path.replace(".pt", "_test_score.json")
        with open(output_path, "w") as fp:
            json.dump(test_summary, fp)

        print(f"Test results saved: {output_path}")
        return test_summary
