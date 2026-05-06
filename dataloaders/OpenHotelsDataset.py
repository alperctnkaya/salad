"""
Dataset classes for the OpenHotels benchmark.

All datasets load images from disk using JSON metadata files.
Transforms are applied inside ``__getitem__`` so DataLoader workers
do the heavy lifting (resize, crop, normalise) in parallel.
"""
import json
import os
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class OpenHotelsDataset(Dataset):
    """
    Loads images from disk using JSON metadata.

    Each metadata entry must have:
        - ``path``:  relative path from *image_root*
        - ``hotel_id``:  hotel identifier (str)

    Returns ``(image_tensor, hotel_id_str)`` per sample.
    """

    def __init__(self, metadata_path: str, image_root: str, transform):
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
        self.image_root = image_root
        self.transform = transform

        # Pre-compute paths and IDs once for fast __getitem__
        self.paths = [
            os.path.join(image_root, entry["path"])
            for entry in self.metadata
        ]
        self.hotel_ids = [str(entry["hotel_id"]) for entry in self.metadata]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        hotel_id = self.hotel_ids[idx]
        try:
            image = Image.open(self.paths[idx]).convert("RGB")
        except (FileNotFoundError, OSError, Image.UnidentifiedImageError):
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        image = self.transform(image)  # PIL → Tensor (runs in worker)
        return image, hotel_id


def tensor_collate(batch):
    """Stack image tensors; keep hotel-ID strings as a list."""
    images, hotel_ids = zip(*batch)
    return torch.stack(images), list(hotel_ids)


class OpenHotelsTrainDataset(Dataset):
    """
    Training dataset that groups images by hotel_id and returns
    ``img_per_place`` images per hotel on each ``__getitem__`` call.

    When a hotel has fewer images than ``img_per_place``, images are
    sampled **with replacement**.  Because the transform pipeline
    includes stochastic augmentations (e.g. ``RandAugment``), each
    re-sampled copy of the same source image becomes a distinct
    augmented view, effectively generating synthetic positives on the
    fly.

    Hotels with fewer than ``min_img_per_place`` images are dropped
    (default ``1`` — keep every hotel).

    Returns:
        images : ``Tensor[K, C, H, W]``
        labels : ``Tensor[K]``  (integer hotel label)
    """

    def __init__(
        self,
        metadata_path: str,
        image_root: str,
        transform,
        img_per_place: int = 4,
        min_img_per_place: int = 1,
        random_sample_from_each_place: bool = True,
    ):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.image_root = image_root
        self.transform = transform
        self.img_per_place = img_per_place
        self.random_sample = random_sample_from_each_place

        # Group indices by hotel_id
        hotel_to_indices: dict[str, list[int]] = defaultdict(list)
        self.all_paths: list[str] = []
        self.all_hotel_ids: list[str] = []

        for idx, entry in enumerate(metadata):
            path = os.path.join(image_root, entry["path"])
            hid = str(entry["hotel_id"])
            self.all_paths.append(path)
            self.all_hotel_ids.append(hid)
            hotel_to_indices[hid].append(idx)

        self.valid_hotel_ids = [
            hid
            for hid, indices in hotel_to_indices.items()
            if len(indices) >= min_img_per_place
        ]
        self.hotel_to_indices = {
            hid: hotel_to_indices[hid] for hid in self.valid_hotel_ids
        }

        self.hotel_to_label = {
            hid: i for i, hid in enumerate(self.valid_hotel_ids)
        }

        self.total_nb_images = sum(
            len(self.hotel_to_indices[hid]) for hid in self.valid_hotel_ids
        )

    def __len__(self):
        return len(self.valid_hotel_ids)

    def _load_image(self, idx: int) -> Image.Image:
        """Load a single PIL image, returning a grey placeholder on error."""
        try:
            return Image.open(self.all_paths[idx]).convert("RGB")
        except (FileNotFoundError, OSError, Image.UnidentifiedImageError):
            return Image.new("RGB", (224, 224), color=(128, 128, 128))

    def __getitem__(self, index):
        hid = self.valid_hotel_ids[index]
        indices = self.hotel_to_indices[hid]
        label = self.hotel_to_label[hid]
        n_available = len(indices)

        # ── Sample K indices ──────────────────────────────────────────
        #  • If enough unique images → sample without replacement
        #  • If too few            → sample WITH replacement so the
        #    same source image is drawn more than once; because the
        #    transform is stochastic (RandAugment etc.) each copy
        #    will be a different augmented view.
        if self.random_sample:
            need_replace = n_available < self.img_per_place
            sampled = np.random.choice(
                indices, self.img_per_place, replace=need_replace
            )
        else:
            # Deterministic: tile the list to reach img_per_place
            repeats = (self.img_per_place // n_available) + 1
            sampled = (indices * repeats)[: self.img_per_place]

        # ── Load & augment ────────────────────────────────────────────
        imgs = [self.transform(self._load_image(idx)) for idx in sampled]

        return torch.stack(imgs), torch.tensor([label] * self.img_per_place)
