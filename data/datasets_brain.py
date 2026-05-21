import cv2
import numpy as np
import torch
from glob import glob
from torch.utils.data import Dataset

from .transforms import get_brain_interactive_transforms


class BrainDataset(Dataset):
    """Brain MRI dataset for binary segmentation (grayscale images)."""

    def __init__(self, images_path, masks_path, transform=False, return_path=False):
        self.images_path = images_path
        self.masks_path = masks_path
        self.return_path = return_path

        if transform == 'interactive':
            self.transform = get_brain_interactive_transforms()
        else:
            self.transform = None

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        image = image[:, :, None]
        mask = np.where(mask == 255, 1, 0).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        mean = np.mean(image)
        std = np.std(image)
        image = ((image - mean) / (std + 1e-8)).astype(np.float32)
        image = torch.from_numpy(np.transpose(image, (2, 0, 1)))

        mask = torch.from_numpy(np.expand_dims(mask.astype(np.int64), axis=0))

        if self.return_path:
            return image, mask, self.images_path[index], self.masks_path[index]
        return image, mask

    def __len__(self):
        return len(self.images_path)


def get_brain_data(img_dir, mask_dir, transform=False, return_path=False):
    
    x = sorted(glob(f"{img_dir}/*"))
    y = sorted(glob(f"{mask_dir}/*"))
    print(f"Loaded {len(x)} images")
    return BrainDataset(x, y, transform=transform, return_path=return_path)
