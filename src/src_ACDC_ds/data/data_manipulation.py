from pathlib import Path
import h5py
import numpy as np
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader

class ACDCSliceDataset(Dataset):
    """Dataset for ACDC cardiac MRI slices formatted for HuggingFace Trainer."""

    def __init__(
        self,
        data_dir: Path,
        file_paths: List[Path],
        target_size: Tuple[int, int] = (256, 256),
        transform: Optional[A.Compose] = None,
        use_scribble: bool = False
    ):
        """
        Args:
            data_dir: Directory containing the h5 slice files
            file_paths: List of h5 slice file paths for a single split
            target_size: Size to resize images to (height, width)
            transform: Optional albumentations transform to apply
            use_scribble: Whether to load scribble annotations (default: False)
        """
        self.data_dir = data_dir
        self.file_paths = file_paths if file_paths else sorted(list(data_dir.glob("patient*_frame*_slice_*.h5")))
        self.target_size = target_size
        self.transform = transform
        self.use_scribble = use_scribble

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset in HuggingFace format.

        Returns:
            dict with keys:
                - pixel_values: Preprocessed image tensor (C, H, W)
                - labels: Label tensor (H, W)
                - path: Original file path
        """
        file_path = self.file_paths[idx]

        with h5py.File(file_path, 'r') as f:
            image = f['image'][()].astype(np.float32)
            label = f['label'][()].astype(np.int64)
            scribble = f['scribble'][()].astype(np.int64) if self.use_scribble else None

            # Normalize image to [0,1]
            image = (image - image.min()) / (image.max() - image.min())

            # Resize if needed
            if image.shape != self.target_size:
                resize = A.Resize(height=self.target_size[0], width=self.target_size[1])
                resized = resize(image=image, mask=label)
                image = resized['image']
                label = resized['mask']
                if self.use_scribble:
                    scribble_resized = resize(image=image, mask=scribble)
                    scribble = scribble_resized['mask']
                    image_copy = image.copy()

            # Apply transforms if any
            if self.transform is not None:
                transformed = self.transform(image=image, mask=label)
                image = transformed['image']
                label = transformed['mask']
                if self.use_scribble:
                    scribble_transformed = self.transform(image=image_copy, mask=scribble)
                    scribble = scribble_transformed['mask']

            label = label.type(torch.int64)

            sample = {
                'pixel_values': image,
                'labels': label,
                'path': str(file_path)
            }

            if self.use_scribble:
                sample['scribble'] = scribble

            return sample
        
def get_transforms(is_training: bool = True) -> A.Compose:
    """
    Get transformation pipeline.
    """
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Normalize(mean=0, std=1),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=0, std=1),
            ToTensorV2()
        ])

def split_patients(
    data_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Split patients into train and validation sets.

    Args:
        data_dir: Directory containing the h5 files
        train_ratio: Ratio of patients for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_patient_ids, val_patient_ids)
    """
    # Extract unique patient IDs
    all_files = list(data_dir.glob("patient*_frame*_slice_*.h5"))
    patient_ids = sorted(list(set(f.name.split('_')[0][7:] for f in all_files)))

    np.random.seed(seed)
    np.random.shuffle(patient_ids)

    split_idx = int(len(patient_ids) * train_ratio)
    return patient_ids[:split_idx], patient_ids[split_idx:]

def cardiac_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for cardiac segmentation data.

    Args:
        batch: List of dictionaries containing:
            - pixel_values: Image tensor (C, H, W)
            - labels: Label tensor (H, W)
            - path: Original file path

    Returns:
        Dictionary with batched tensors:
            - pixel_values: (B, C, H, W)
            - labels: (B, H, W)
            - paths: List of file paths
    """
    paths = [item.pop('path') for item in batch]

    batched_data = default_collate(batch)

    batched_data['paths'] = paths

    assert batched_data['pixel_values'].ndim == 4, "Wrong dimensions for pixel_values"
    assert batched_data['labels'].ndim == 3, "Wrong dimensions for labels"

    return batched_data

def create_dataloaders(
    data_dir: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    target_size: Tuple[int, int] = (256, 256)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing the h5 files
        batch_size: Batch size for both loaders
        num_workers: Number of workers for data loading
        train_ratio: Ratio of patients for training
        target_size: Size to resize images to (height, width)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_patients, val_patients = split_patients(data_dir, train_ratio)
    print(f"Number of patients in training: {len(train_patients)}")
    print(f"Number of patients in validation: {len(val_patients)}")

    train_files = [
        f for f in data_dir.glob("patient*_frame*_slice_*.h5")
        if f.name.split('_')[0][7:] in train_patients
    ]
    val_files = [
        f for f in data_dir.glob("patient*_frame*_slice_*.h5")
        if f.name.split('_')[0][7:] in val_patients
    ]

    print(f"Number of slices in training: {len(train_files)}")
    print(f"Number of slices in validation: {len(val_files)}")

    # Create datasets
    train_dataset = ACDCSliceDataset(
        data_dir=data_dir,
        file_paths=train_files,
        target_size=target_size,
        transform=get_transforms(is_training=True),
        use_scribble=False
    )

    val_dataset = ACDCSliceDataset(
        data_dir=data_dir,
        file_paths=val_files,
        target_size=target_size,
        transform=get_transforms(is_training=False),
        use_scribble=False
    )

    # Create dataloaders with custom collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=cardiac_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=cardiac_collate_fn
    )

    return train_loader, val_loader