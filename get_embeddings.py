from torch.hub import load
from model import dino_backbones
from PIL import Image
import os
from torchvision import transforms
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

backbone_name = 'dinov2_b'

patch_size = dino_backbones[backbone_name]['patch_size']
embedding_size = dino_backbones[backbone_name]['embedding_size']
backbone = load('facebookresearch/dinov2', dino_backbones[backbone_name]['name'])
backbone.eval()
backbone.to(device)
backone = torch.compile(backbone)

base_size = (24, 24)
img_transform = transforms.Compose([
    transforms.Resize((patch_size*base_size[0], patch_size*base_size[1]), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image_path = './data/train'
save_path = './data/testing_here'
os.makedirs(save_path, exist_ok=True)

# for file in tqdm.tqdm(sorted(os.listdir(image_path)), total=len(os.listdir(image_path)), desc="Encoding images"):
#     if f"{save_path}/{file.split('.')[0]}.pt" in os.listdir(save_path):
#         continue
#     base_file_name = file.split(".")[0]
#     img_path = os.path.join(image_path, file)
#     image = Image.open(img_path).convert("RGB")
#     im_tensor = img_transform(image).unsqueeze(0)
#     with torch.no_grad():
#         embedding = backbone(im_tensor.to(device))
#         embedding = embedding.squeeze().cpu()
#     # save torch embedding
#     torch.save(embedding, f"{save_path}/{base_file_name}.pt")


class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, image_path, transform=None):
        self.samples = []
        for file in os.listdir(image_path):
            base_file_name = file.split(".")[0]
            img_path = os.path.join(image_path, file)
            self.samples.append((img_path, base_file_name))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx][0]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


N = 2
dataset = CustomImageFolder(image_path, transform=img_transform)
data_loader = DataLoader(dataset, batch_size=N, shuffle=False, pin_memory=True, drop_last=False, num_workers=8)

from torch.cuda.amp import autocast

# Iterate over the data loader
for i, x in enumerate(tqdm.tqdm(data_loader, total=len(data_loader), desc="Encoding images")):
    batch_size = x.shape[0]
    mask_dim = (x.shape[2] / patch_size, x.shape[3] / patch_size) 
    with torch.no_grad():
        with autocast():
            # x = backbone(x.to(device))
            x = backbone.forward_features(x.to(device))
        x = x['x_norm_patchtokens']
        # x = x.permute(0, 2, 1)
        # # x = x.reshape(batch_size, embedding_size, int(mask_dim[0]), int(mask_dim[1]))
        x = x.detach().cpu()

    # Save each embedding in the batch
    for j in range(x.shape[0]):
        base_file_name = dataset.samples[i*N+j][0].split("/")[-1].split(".")[0]
        print(x[j].shape)
        torch.save(x[j], f"{save_path}/{base_file_name}.pt")
