from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import cv2


def rescale_and_pad(image, target_size=(512, 512), pad_value=0):
    """
    This function rescales the image while maintaining the aspect ratio and pads the image to make it target_sizextarget_size
    Args:
        image: np.ndarray, the image to be rescaled and padded
        target_size: tuple, the target size for the rescaled and padded image
        pad_value: int, the value to be used for padding
    Returns:
        canvas: np.ndarray, the rescaled and padded image
    """
    # Get the original dimensions of the image
    height, width, _ = image.shape
    # Calculate the scaling factor for width and height to make the longest side target_size
    scale_factor = max(target_size) / max(height, width)
    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    # Get the new dimensions after resizing
    new_height, new_width, _ = resized_image.shape
    # Create a new canvas of size target_sizextarget_size
    canvas = np.ones((*target_size, 3), dtype=np.uint8) * pad_value
    # Calculate the padding needed for both height and width
    pad_height = (target_size[0] - new_height) // 2
    pad_width = (target_size[1] - new_width) // 2
    # Paste the resized image onto the canvas
    canvas[pad_height:pad_height + new_height, pad_width:pad_width + new_width, :] = resized_image
    return canvas


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, img_transform=None, mask_transform=None, images=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

        # Only include images for which a mask is found
        if images is None:
            self.images = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + ".png"))]
        else:
            self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx].split(".")[0]
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, img_name + ".png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = (self.mask_transform(mask) * 255).long()

        mask, _ = torch.max(mask, dim=0)
        # mask[mask == (self.num_classes - 1)] == -1
        # bin_mask = torch.zeros(self.num_classes, mask.shape[0], mask.shape[1])
        # for i in range(self.num_classes):
            # bin_mask[i] = (mask == i).float()  # Ensure resulting mask is float type
        return image, mask


class LatentDataset(Dataset):
    def __init__(self, latent_dir, mask_dir, num_classes, img_transform=None, mask_transform=None, tensors=None):
        self.latent_dir = latent_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

        # Only include images for which a mask is found
        if tensors is None:
            self.tensors = [latent for latent in os.listdir(latent_dir) if os.path.isfile(os.path.join(mask_dir, latent.split(".")[0] + ".png"))]
        else:
            self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        img_name = self.tensors[idx].split(".")[0]
        latent = torch.load(os.path.join(self.mask_dir, img_name + ".pt"))
        mask_path = os.path.join(self.mask_dir, img_name + ".png")
        mask = Image.open(mask_path)
        if self.mask_transform:
            mask = (self.mask_transform(mask) * 255).long()
        mask, _ = torch.max(mask, dim=0)
        return latent, mask