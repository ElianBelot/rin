import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CelebA


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int = 64,
        batch_size: int = 256,
        num_workers: int = 0,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ):
        """PyTorch Lightning DataModule for CelebA.

        Parameters:
        - data_dir: path to store/download dataset
        - image_size: final size of square image (e.g. 64)
        - batch_size: number of images per batch
        - num_workers: dataloader worker threads
        - max_train_samples: max number of training samples to use
        - max_val_samples: max number of validation samples to use
        - persistent_workers: whether to maintain worker processes between batches
        - pin_memory: whether to pin memory in data transfer to GPU
        """
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        # Download only once
        CelebA(self.data_dir, split="train", download=True)

    def setup(self, stage: str | None = None):
        transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),  # [0, 1]
                transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
            ]
        )

        train = CelebA(root=self.data_dir, split="train", transform=transform)
        val = CelebA(root=self.data_dir, split="valid", transform=transform)

        self.train_dataset = Subset(train, list(range(self.max_train_samples))) if self.max_train_samples else train
        self.val_dataset = Subset(val, list(range(self.max_val_samples))) if self.max_val_samples else val

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )
