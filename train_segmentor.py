import torch
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from model import Segmentor
from tools.segmentation import SegmentationDataset, train, validation, infer
import wandb
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

wandb.init(project="clothing-seg")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

base_size = (64, 64)
img_transform = transforms.Compose([
    transforms.Resize((14*base_size[0], 14*base_size[1]), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((base_size[0]*4, base_size[1]*4), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


num_classes = 47  # Ignore background so 0 to 45
dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes=num_classes, img_transform=img_transform, mask_transform=mask_transform)

# Splitting data into train and validation sets
train_imgs, valid_imgs = train_test_split(dataset.images, test_size=0.2, random_state=42)
# valid_imgs, test_imgs = train_test_split(valid_imgs, test_size=0.999, random_state=42)
print(len(train_imgs))

train_dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes=num_classes, img_transform=img_transform, mask_transform=mask_transform, images=train_imgs)
valid_dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes=num_classes, img_transform=img_transform, mask_transform=mask_transform, images=valid_imgs)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=8)
valid_loader = DataLoader(valid_dataset, batch_size=16, pin_memory=True, num_workers=8)

model = Segmentor(num_classes, backbone='dinov2_l')
model = model.to(device)

optimizer = optim.Adam(model.trainable_parameters(), lr=3e-4)
weights = torch.ones(num_classes)
weights[-1] = 0.3  # Reduce the weight for the background class
criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))

num_epochs = 2
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch, device)
    validation(model, criterion, valid_loader, device)
    torch.save(model.state_dict(), f'weights/segmentation_model_dinol_{epoch}.pt')

# load state dict
model.load_state_dict(torch.load(f'weights/segmentation_model_dinol_{epoch}.pt'))

# now infer on a random image from the dataset
img = "/home/ubuntu/Downstream-Dinov2/data/train/0a0a539316af6547b3bbe228ead13730.jpg"
mask_label = "/home/ubuntu/Downstream-Dinov2/data/masks/0a0a539316af6547b3bbe228ead13730.png"
mask_label = (mask_transform(Image.open(mask_label)) * 255).cpu().squeeze().permute(1, 2, 0).numpy().astype(np.uint8)[:, :, 0]
img, mask = infer(img, model, device, img_transform)
# they are numpy arrays
print(img.shape, mask.shape, mask_label.shape)
plt.imsave("infered_mask.png", mask.astype(np.uint8))
img = np.clip(img, 0, 1)
plt.imsave("original_img.png", img)
plt.imsave("mask_label.png", mask_label)
