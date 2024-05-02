import torch
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from model import Segmentor
from tools.segmentation import SegmentationDataset, train, validation, infer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

base_size = (32, 32)
img_transform = transforms.Compose([
    transforms.Resize((14*base_size[0],14*base_size[1]), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((base_size[0]*4,base_size[1]*4), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


num_classes = 47
dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes = num_classes, img_transform=img_transform, mask_transform=mask_transform)

# Splitting data into train and validation sets
train_imgs, valid_imgs = train_test_split(dataset.images, test_size=0.99, random_state=42)
# valid_imgs, test_imgs = train_test_split(valid_imgs, test_size=0.999, random_state=42)
print(len(train_imgs))

train_dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes = num_classes, img_transform=img_transform, mask_transform=mask_transform, images=train_imgs)
valid_dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes = num_classes, img_transform=img_transform, mask_transform=mask_transform, images=valid_imgs)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
valid_loader = DataLoader(train_dataset, batch_size=8, pin_memory=True)

model = Segmentor(num_classes)
model = model.to(device)

optimizer = optim.Adam(model.trainable_parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch, device)
    validation(model, criterion, valid_loader, device)

torch.save(model.state_dict(), 'weights/segmentation_model_2.pt')

