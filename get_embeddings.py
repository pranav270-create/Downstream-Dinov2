from torch.hub import load
from model import dino_backbones

backbone = 'dinov2_s'
backbone = load('facebookresearch/dinov2', dino_backbones[backbone]['name'])
backbone.eval()

embedding = backbone()