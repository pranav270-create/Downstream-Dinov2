import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import Segmentor
from train_test_loop import infer

device = torch.device('cuda')

model = Segmentor(47, backbone='dinov2_l', head='depth_conv')
model = model.to(device)

loadpath = 'weights/mosaic_seg_d_conv_lg.pt'
torch_dict = torch.load(loadpath)
model.load_state_dict(torch_dict)

input_size = (32, 32)
img_transform = transforms.Compose([
    transforms.Resize((14*input_size[0], 14*input_size[1])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_size = (input_size[0] * 8, input_size[1] * 8)
mask_transform = transforms.Compose([
    transforms.Resize((mask_size[0], mask_size[1]), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
])

img = Image.open("original_img.png").convert("RGB")
base_shape = np.array(img).shape

inverse_transform = transforms.Compose([
    transforms.Resize((base_shape[0], base_shape[1]), interpolation=transforms.InterpolationMode.NEAREST),
])

# now infer on a random image from the dataset
import os
train_path = "/home/ubuntu/Downstream-Dinov2/data/train"
save_path = "/home/ubuntu/Downstream-Dinov2/examples"
count = 0
for imfile in sorted(os.listdir(train_path)):
    if count > 10:
        break
    basefile = imfile.split(".")[0]
    img = f"/home/ubuntu/Downstream-Dinov2/data/train/{basefile}.jpg"
    mask_label = f"/home/ubuntu/Downstream-Dinov2/data/masks/{basefile}.png"
    mask_label = (mask_transform(Image.open(mask_label)) * 255).cpu().squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
    img, mask = infer(img, model, device, img_transform)
    mask = np.transpose(mask, (1, 2, 0))
    mask = np.repeat(mask, 3, -1)
    plt.imsave(f"{save_path}/predicted/{basefile}.png", mask.astype(np.uint8))
    # plt.imsave(f"{save_path}/masks/{basefile}.png", mask_label)
    count += 1
