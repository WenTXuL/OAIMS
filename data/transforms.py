import cv2
import albumentations as A
from albumentations import (
    CLAHE, ColorJitter, Equalize, FancyPCA, HueSaturationValue,
    ISONoise, MultiplicativeNoise, RandomBrightnessContrast,
    RandomGamma, GaussNoise, Blur, MedianBlur, RGBShift,
)


def get_interactive_transforms(p=0.8):
    return A.Compose([
        A.ShiftScaleRotate(
            p=0.1, shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0,
        ),
        A.ElasticTransform(
            p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0,
            alpha_affine=30, mask_value=0,
        ),
        A.Rotate(limit=270, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        CLAHE(p=0.1),
        ColorJitter(p=0.2),
        Equalize(p=0.05),
        FancyPCA(p=0.1),
        HueSaturationValue(p=0.1),
        ISONoise(p=0.1),
        MultiplicativeNoise(p=0.1),
        RandomBrightnessContrast(p=0.3),
        RandomGamma(p=0.2),
        GaussNoise(p=0.1),
        Blur(p=0.1),
        MedianBlur(p=0.01),
        RGBShift(p=0.05),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.05),
    ], p=p)


def get_brain_interactive_transforms(p=0.8):
    """Transforms for brain MRI (grayscale). No colour-specific augmentations."""
    return A.Compose([
        A.ShiftScaleRotate(
            p=0.1, shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0,
        ),
        A.ElasticTransform(
            p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0,
            alpha_affine=30, mask_value=0,
        ),
        A.Rotate(limit=270, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        CLAHE(p=0.1),
        MultiplicativeNoise(p=0.1),
        RandomBrightnessContrast(p=0.3),
        RandomGamma(p=0.2),
        GaussNoise(p=0.1),
        Blur(p=0.1),
        MedianBlur(p=0.01),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.05),
    ], p=p)
