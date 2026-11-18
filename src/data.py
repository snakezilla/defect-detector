"""Load MVTec AD images and defect masks for one category."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 256
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

IMAGE_TF = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
MASK_TF = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.NEAREST),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
])


@dataclass(frozen=True)
class Sample:
    image: torch.Tensor       # 3 x 224 x 224, normalized
    mask: torch.Tensor        # 1 x 224 x 224, 1 = defect pixel
    label: int                # 0 = good, 1 = defect
    defect_type: str
    path: str


def _load_image(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        return IMAGE_TF(im.convert("RGB"))


def _load_mask(path: Path | None) -> torch.Tensor:
    if path is None:
        return torch.zeros(1, CROP_SIZE, CROP_SIZE)
    with Image.open(path) as im:
        return (MASK_TF(im.convert("L")) > 0.5).float()


def train_images(root: Path, category: str) -> list[Path]:
    folder = root / category / "train" / "good"
    if not folder.is_dir():
        raise FileNotFoundError(f"missing training folder: {folder}")
    return sorted(folder.glob("*.png"))


def test_samples(root: Path, category: str) -> list[Sample]:
    test_dir = root / category / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"missing test folder: {test_dir}")
    samples = []
    for type_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        for img_path in sorted(type_dir.glob("*.png")):
            is_good = type_dir.name == "good"
            mask_path = None if is_good else root / category / "ground_truth" / type_dir.name / f"{img_path.stem}_mask.png"
            samples.append(Sample(
                image=_load_image(img_path),
                mask=_load_mask(mask_path),
                label=0 if is_good else 1,
                defect_type=type_dir.name,
                path=str(img_path),
            ))
    return samples


def load_batch(paths: list[Path]) -> torch.Tensor:
    return torch.stack([_load_image(p) for p in paths])


def denormalize(image: torch.Tensor) -> np.ndarray:
    """Return an H x W x 3 array in 0..1 for plots."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
