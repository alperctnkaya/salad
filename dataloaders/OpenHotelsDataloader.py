"""
DataModule that wraps the local-disk OpenHotels datasets for training and
evaluation.

Usage::

    dm = OpenHotelsDataModule(batch_size=32)
    dm.setup()
    train_loader = dm.train_dataloader()
"""
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .OpenHotelsDataset import OpenHotelsTrainDataset, OpenHotelsDataset, tensor_collate
from .config import GALLERY_METADATA, TEST_OBJECT_METADATA, TEST_NON_OBJECT_METADATA, IMAGE_ROOT

# Standard ImageNet normalisation
IMAGENET_MEAN_STD = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


class OpenHotelsDataModule:
    """
    Provides train / eval DataLoaders backed by local JSON metadata and
    on-disk images.

    Args:
        batch_size:       Mini-batch size (number of *hotels* per batch for
                          training; number of *images* for eval).
        img_per_place:    Images sampled per hotel in each training step.
        min_img_per_place: Minimum images a hotel must have to be included.
        shuffle_all:      Shuffle the training DataLoader.
        image_size:       Resize target ``(H, W)``.
        num_workers:      DataLoader workers.
        mean_std:         Normalisation statistics.
        random_sample_from_each_place: Randomly pick K images per hotel.
        gallery_metadata: Path to gallery JSON (default from config).
        test_object_metadata: Path to test-object JSON.
        test_non_object_metadata: Path to test-non-object JSON.
        image_root:       Root directory for images.
    """

    def __init__(
        self,
        batch_size: int = 32,
        img_per_place: int = 4,
        min_img_per_place: int = 4,
        shuffle_all: bool = False,
        image_size: tuple = (224, 224),
        num_workers: int = 4,
        mean_std: dict = IMAGENET_MEAN_STD,
        random_sample_from_each_place: bool = True,
        gallery_metadata: str = GALLERY_METADATA,
        test_object_metadata: str = TEST_OBJECT_METADATA,
        test_non_object_metadata: str = TEST_NON_OBJECT_METADATA,
        image_root: str = IMAGE_ROOT,
    ):
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.random_sample_from_each_place = random_sample_from_each_place

        # Paths
        self.gallery_metadata = gallery_metadata
        self.test_object_metadata = test_object_metadata
        self.test_non_object_metadata = test_non_object_metadata
        self.image_root = image_root

        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]

        # ── Transforms ────────────────────────────────────────────────────
        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.eval_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        # Lazy-init
        self.train_dataset = None

    # ── Setup ─────────────────────────────────────────────────────────────

    def setup(self):
        """Initialise the training dataset (gallery images grouped by hotel)."""
        self.train_dataset = OpenHotelsTrainDataset(
            metadata_path=self.gallery_metadata,
            image_root=self.image_root,
            transform=self.train_transform,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
        )

    # ── DataLoaders ───────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_all,
            drop_last=False,
            pin_memory=True,
        )

    def gallery_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Flat gallery DataLoader for evaluation (one image per sample)."""
        ds = ImageDataset(
            metadata_path=self.gallery_metadata,
            image_root=self.image_root,
            transform=self.eval_transform,
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size or self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=tensor_collate,
        )

    def test_object_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Test-object query DataLoader (one image per sample)."""
        ds = ImageDataset(
            metadata_path=self.test_object_metadata,
            image_root=self.image_root,
            transform=self.eval_transform,
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size or self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=tensor_collate,
        )

    def test_non_object_dataloader(self, batch_size: int | None = None) -> DataLoader:
        """Test-non-object query DataLoader (one image per sample)."""
        ds = ImageDataset(
            metadata_path=self.test_non_object_metadata,
            image_root=self.image_root,
            transform=self.eval_transform,
        )
        return DataLoader(
            dataset=ds,
            batch_size=batch_size or self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=tensor_collate,
        )
