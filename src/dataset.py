from __future__ import annotations

import pickle
from dataclasses import dataclass

import h5py
import torch
from torch.utils.data import Dataset
from turbojpeg import TJFLAG_FASTDCT, TJFLAG_FASTUPSAMPLE, TJPF_RGB, TurboJPEG


@dataclass
class PatchFeatureDataset(Dataset):
    """A whole-slide-image dataset where each item is a bag of patches.

    Two input formats are supported:
      * ``.h5``  -- precomputed patch feature vectors.
      * ``.pkl`` -- raw JPEG bytes of the patches (for end-to-end training).
    """

    filenames: list[str]
    labels: list[int]
    column: str = "20.0x_patches"
    max_length: int = -1

    def __len__(self) -> int:
        return len(self.filenames)

    def _read_features_from_h5(self, filename: str) -> torch.Tensor:
        with h5py.File(filename, "r") as fp:
            output = torch.as_tensor(fp[self.column][:], dtype=torch.float32)
        if self.max_length > 0:
            output = output[: self.max_length]
        return output

    def _read_bytes_from_pkl(self, filename: str) -> list[bytes]:
        with open(filename, "rb") as fp:
            output = pickle.load(fp)[self.column]
        if self.max_length > 0:
            output = output[: self.max_length]
        return output

    def __getitem__(self, idx: int) -> tuple[torch.Tensor | list[bytes], torch.Tensor]:
        if (filename := self.filenames[idx]).endswith(".h5"):
            return self._read_features_from_h5(filename), self.labels[idx]
        elif filename.endswith(".pkl"):
            return self._read_bytes_from_pkl(filename), self.labels[idx]
        raise NotImplementedError(f"{filename.split('.')[-1]} is not supported.")


@dataclass
class DecodeJPEGBytesDataset(Dataset):
    """Decodes raw JPEG bytes into RGB tensors on the fly using TurboJPEG."""

    bytes_list: list[str]
    jpeg: TurboJPEG = TurboJPEG()

    def __len__(self) -> int:
        return len(self.bytes_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = self.jpeg.decode(
            self.bytes_list[idx][0],
            pixel_format=TJPF_RGB,
            flags=TJFLAG_FASTUPSAMPLE | TJFLAG_FASTDCT,
        )
        return torch.as_tensor(img)
