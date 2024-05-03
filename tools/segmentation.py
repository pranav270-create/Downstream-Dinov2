from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import tqdm
import wandb
import numpy as np


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
        # bin_mask = torch.zeros(self.num_classes, mask.shape[0], mask.shape[1])
        # for i in range(self.num_classes):
            # bin_mask[i] = (mask == i).float()  # Ensure resulting mask is float type
        return image, mask


from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


def train(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0

    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc="Training Loop"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # output = model(data)
        # torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 2.0)
        # loss = criterion(output, target)
        # loss.backward()
        # optimizer.step()
        running_loss += loss.item()
        wandb.log({"train_batch_loss": loss.item()})

    wandb.log({"train_epoch_loss": running_loss/len(train_loader)})
    print(f'\nTrain set: Average loss: {running_loss/len(train_loader):.6f}')



def calculate_miou(target, predicted):
    intersection = np.logical_and(target, predicted)
    union = np.logical_or(target, predicted)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def validation(model, criterion, valid_loader, device):
    model.eval()
    running_loss = 0
    correct = 0
    total_iou = 0

    with torch.no_grad():
        for data, target in tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Validation Loop"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()
            # Calculate mIoU
            iou_score = calculate_miou(target.cpu().numpy(), predicted.cpu().numpy())
            total_iou += iou_score

    wandb.log({"val_loss": running_loss/len(valid_loader)})
    wandb.log({"val_accuracy": correct/len(valid_loader)})
    wandb.log({"val_mIoU": total_iou/len(valid_loader)})
    print(f'\nValidation set: Average loss: {running_loss/len(valid_loader):.6f}')


def infer(image_path, model, device, img_transform):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = img_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make sure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Make prediction
        output = model(transformed_image)

        # Get the predicted class for each pixel
        _, predicted = torch.max(output, 1)

    # Move prediction to cpu and convert to numpy array
    predicted = predicted.squeeze().cpu().numpy()

    return transformed_image.cpu().squeeze().permute(1, 2, 0).numpy(), predicted
