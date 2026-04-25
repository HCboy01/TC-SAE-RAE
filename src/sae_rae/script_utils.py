from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


def add_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def safe_torch_load(path: str | Path, map_location: str = "cpu"):
    load_sig = inspect.signature(torch.load)
    kwargs = {"map_location": map_location}
    if "mmap" in load_sig.parameters:
        kwargs["mmap"] = True
    if "weights_only" in load_sig.parameters:
        kwargs["weights_only"] = True
    return torch.load(path, **kwargs)


def full_torch_load(path: str | Path, map_location: str = "cpu"):
    load_sig = inspect.signature(torch.load)
    kwargs = {"map_location": map_location}
    if "mmap" in load_sig.parameters:
        kwargs["mmap"] = True
    if "weights_only" in load_sig.parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


class RecursiveImageDataset(Dataset):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str,
        transform=None,
        *,
        return_path: bool = False,
        max_samples: int | None = None,
        files: list[Path] | None = None,
        label=0,
    ):
        self.root = Path(root)
        self.transform = transform
        self.return_path = bool(return_path)
        self.label = label
        self.files = list(files) if files is not None else self.collect_files(self.root)
        if max_samples is not None:
            self.files = self.files[:max_samples]
        if not self.files:
            raise RuntimeError(f"No images found under: {self.root}")

    @classmethod
    def collect_files(cls, root: Path) -> list[Path]:
        if not root.exists():
            raise FileNotFoundError(f"Data path does not exist: {root}")
        return sorted(
            p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in cls.IMG_EXTS
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.return_path:
            return image, str(path)
        return image, self.label
