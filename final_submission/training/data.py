from __future__ import annotations

import json
from pathlib import Path

from torchvision import transforms

from dataset import PokemonDataset, read_split


class TrainImages(PokemonDataset):
    """Thin train-split-only adapter around FastGen's protected PokemonDataset."""

    def __init__(self, data_root: str | Path, split_dir: str | Path, image_size: int, num_classes: int | None = None,
                 horizontal_flip_probability: float = 0.0, augmentation_degrees: float = 0.0,
                 augmentation_translate: float = 0.0, augmentation_scale_delta: float = 0.0,
                 split_file: str | Path | None = None, cache: bool = False):
        split_dir = Path(split_dir)
        mapping = json.loads((split_dir / "category_to_id.json").read_text(encoding="utf-8"))
        if num_classes is not None and len(mapping) != num_classes:
            raise ValueError(f"Found {len(mapping)} classes, config requires {num_classes}")
        selected_split = Path(split_file) if split_file else split_dir / "train_split.txt"
        selected_paths = read_split(selected_split)
        if not selected_paths or len(set(selected_paths)) != len(selected_paths):
            raise ValueError("Training manifest must be nonempty and contain unique paths")
        if not set(selected_paths) <= set(read_split(split_dir / "train_split.txt")):
            raise ValueError("Manifest must contain only official training paths")
        operations = [transforms.Resize((image_size, image_size))]
        if horizontal_flip_probability:
            operations.append(transforms.RandomHorizontalFlip(horizontal_flip_probability))
        if augmentation_degrees or augmentation_translate or augmentation_scale_delta:
            operations.append(transforms.RandomAffine(
                degrees=augmentation_degrees,
                translate=(augmentation_translate, augmentation_translate),
                scale=(1 - augmentation_scale_delta, 1 + augmentation_scale_delta),
                interpolation=transforms.InterpolationMode.BILINEAR, fill=(255, 255, 255),
            ))
        operations.extend([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        super().__init__(
            data_root=data_root,
            split_file=selected_split,
            transform=transforms.Compose(operations),
            return_category=True,
            category_to_id=mapping,
        )
        self.classes = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
        if cache and (horizontal_flip_probability or augmentation_degrees or augmentation_translate or augmentation_scale_delta):
            raise ValueError("Image caching requires augmentation disabled")
        self.cached_samples = [super(TrainImages,self).__getitem__(i) for i in range(len(self))] if cache else None

    def __getitem__(self, index: int):
        if self.cached_samples is not None:
            return self.cached_samples[index]
        return super().__getitem__(index)
