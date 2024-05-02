import torch
import torch.nn as nn
from torch.hub import load


dino_backbones = {
    'dinov2_s':{
        'name':'dinov2_vits14',
        'embedding_size':384,
        'patch_size':14
    },
    'dinov2_b':{
        'name':'dinov2_vitb14',
        'embedding_size':768,
        'patch_size':14
    },
    'dinov2_l':{
        'name':'dinov2_vitl14',
        'embedding_size':1024,
        'patch_size':14
    },
    'dinov2_g':{
        'name':'dinov2_vitg14',
        'embedding_size':1536,
        'patch_size':14
    },
}


class linear_head(nn.Module):
    def __init__(self, embedding_size = 384, num_classes = 5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class conv_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=47):
        super(conv_head, self).__init__()
        hidden_layer_size = 128
        self.segmentation_conv = nn.Sequential(
            # nn.GroupNorm(32, embedding_size),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(embedding_size, hidden_layer_size, (3, 3), padding=(1, 1)),
            # nn.GroupNorm(32, hidden_layer_size),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_layer_size, num_classes, (3, 3), padding=(1, 1)),
        )
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.segmentation_conv(x)
        x = torch.sigmoid(x)  # batch_size x num_classes x H x W
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s', head = 'linear', backbones = dino_backbones):
        super(Classifier, self).__init__()
        self.heads = {
            'linear':linear_head
        }
        self.backbones = dino_backbones
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.backbone.eval()
        self.head = self.heads[head](self.backbones[backbone]['embedding_size'],num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x


class Segmentor(nn.Module):
    def __init__(self, num_classes, backbone = 'dinov2_s', head = 'conv', backbones = dino_backbones):
        super(Segmentor, self).__init__()
        self.heads = {
            'conv': conv_head
        }
        self.backbones = dino_backbones
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.backbone.eval()
        self.num_classes =  num_classes # add a class for background if needed
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']
        self.head = self.heads[head](self.embedding_size,self.num_classes)
        self.trainable_parameters = self.head.parameters

    def forward(self, x):
        batch_size = x.shape[0]
        mask_dim = (x.shape[2] / self.patch_size, x.shape[3] / self.patch_size) 
        with torch.no_grad():
            x = self.backbone.forward_features(x.cuda())
            x = x['x_norm_patchtokens']
            x = x.permute(0, 2, 1)
            x = x.reshape(batch_size, self.embedding_size, int(mask_dim[0]), int(mask_dim[1]))
        x = self.head(x)
        return x



