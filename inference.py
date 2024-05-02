import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

from model import Segmentor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Segmentor(91)
model = model.to(device)
input_size = (32, 60)
img_transform = transforms.Compose([
    transforms.Resize((14*input_size[0],14*input_size[1])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_size = (input_size[0] * 4, input_size[1] * 4)
mask_transform = transforms.Compose([
    transforms.Resize((mask_size[0], mask_size[1]), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])


img = Image.open("examples/test.jpg").convert("RGB")
mask = img
base_shape = np.array(img).shape

inverse_transform = transforms.Compose([
    transforms.Resize((base_shape[0], base_shape[1]), interpolation=transforms.InterpolationMode.NEAREST),
])

print(f"Original Image: {base_shape}")
transform_img = img_transform(img).unsqueeze(0).to(device)
print(f"Transformed Image: {transform_img.shape}")
out = model(transform_img)
print(f"Segmentation Mask: {out.shape}")
_, predicted = torch.max(out, 1)
print(f"Softmax: {predicted.shape}")
transformed_mask = mask_transform(mask).to(device)
print(f"Model Prediction: {transformed_mask.shape}")
masked_output = inverse_transform(transformed_mask)
print(f"Masked Output: {masked_output.shape}")
