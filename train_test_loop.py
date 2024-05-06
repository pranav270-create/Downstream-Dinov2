from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import tqdm
import wandb
import numpy as np
import evaluate


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


metric = evaluate.load("mean_iou")

def compute_metrics(logits, labels, num_labels, ignore_index):
    with torch.no_grad():
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)

        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=ignore_index,
            reduce_labels=False,
        )
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics[key] = value.tolist()
        return metrics


def batch_pix_accuracy(output, target, ignore_index):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1
    mask = (target != (ignore_index + 1))

    pixel_labeled = torch.sum(mask).item()
    pixel_correct = torch.sum((predict == target) * mask).item()
    # assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


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
            # Calculate mIoU
            iou_score = compute_metrics(output, target, num_labels=47, ignore_index=46)['mean_iou']
            total_iou += iou_score
            pix_acc = batch_pix_accuracy(output, target, ignore_index=46)
            correct += pix_acc[0] / pix_acc[1]

        wandb.log({"val_loss": running_loss/len(valid_loader)})
        wandb.log({"val_accuracy": correct/len(valid_loader)})
        wandb.log({"val_mIoU": total_iou/len(valid_loader)})
    print(f'\nValidation set: Average loss: {running_loss/len(valid_loader):.6f}')


def infer(image_path, model, device, img_transform):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = img_transform(image).unsqueeze(0).to(device)

    # Make sure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Make prediction
        output = model(transformed_image)

        # Get the predicted class for each pixel
        _, predicted = torch.max(output, 1)

    # Move prediction to cpu and convert to numpy array
    predicted = predicted.cpu().numpy()

    return transformed_image.cpu().squeeze().permute(1, 2, 0).numpy(), predicted