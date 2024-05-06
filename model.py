import torch
import torch.nn as nn
from torch.hub import load


dino_backbones = {
    'dinov2_s': {
        'name': 'dinov2_vits14_reg',
        'embedding_size': 384,
        'patch_size': 14
    },
    'dinov2_b': {
        'name': 'dinov2_vitb14_reg',
        'embedding_size': 768,
        'patch_size': 14
    },
    'dinov2_l': {
        'name': 'dinov2_vitl14_reg',
        'embedding_size': 1024,
        'patch_size': 14
    },
    'dinov2_g': {
        'name': 'dinov2_vitg14_reg',
        'embedding_size': 1536,
        'patch_size': 14
    },
}


class linear_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=5):
        super(linear_head, self).__init__()
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class conv_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=47):
        super(conv_head, self).__init__()
        hidden_layer_size = 256
        self.segmentation_conv = nn.Sequential(
            nn.GroupNorm(32, embedding_size),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(embedding_size, hidden_layer_size, (3, 3), padding=(1, 1)),
            nn.GroupNorm(32, hidden_layer_size),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_layer_size, hidden_layer_size*2, (3, 3), padding=(1, 1)),
            nn.GroupNorm(32, hidden_layer_size*2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_layer_size*2, num_classes, (3, 3), padding=(1, 1)),
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
        # x = torch.softmax(x, dim=1)
        return x


class depth_conv_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=47):
        super(depth_conv_head, self).__init__()
        hidden_layer_size = 4096
        self.segmentation_conv = nn.Sequential(
            nn.GroupNorm(32, embedding_size),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(embedding_size, embedding_size, kernel_size=3, padding=1, groups=embedding_size),
            nn.Conv2d(embedding_size, hidden_layer_size, kernel_size=1),
            nn.GroupNorm(32, hidden_layer_size),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_layer_size, hidden_layer_size, kernel_size=3, padding=1, groups=hidden_layer_size),
            nn.Conv2d(hidden_layer_size, hidden_layer_size//2, kernel_size=1),
            nn.GroupNorm(32, hidden_layer_size//2),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(hidden_layer_size//2, hidden_layer_size//2, kernel_size=3, padding=1, groups=hidden_layer_size//2),
            nn.Conv2d(hidden_layer_size//2, num_classes, kernel_size=1),
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
        # x = torch.softmax(x, dim=1)
        return x

    
class residual_depth_conv_head(nn.Module):
    def __init__(self, embedding_size=384, num_classes=47):
        super(residual_depth_conv_head, self).__init__()
        hidden_layer_size = 4096
        last_hidden_layer_size = 1024

        self.groupnorm1 = nn.GroupNorm(32, embedding_size)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1_a = nn.Conv2d(embedding_size, embedding_size, kernel_size=3, padding=1, groups=embedding_size)
        self.conv1_b = nn.Conv2d(embedding_size, hidden_layer_size, kernel_size=1)
        # self.residual_conv_1 = nn.Conv2d(embedding_size, hidden_layer_size, kernel_size=1)

        self.groupnorm2 = nn.GroupNorm(32, hidden_layer_size)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv2_a = nn.Conv2d(hidden_layer_size, hidden_layer_size, kernel_size=3, padding=1, groups=hidden_layer_size)
        self.conv2_b = nn.Conv2d(hidden_layer_size, hidden_layer_size, kernel_size=1)
        # self.residual_conv_2 = nn.Conv2d(hidden_layer_size, hidden_layer_size, kernel_size=1)

        self.groupnorm3 = nn.GroupNorm(32, hidden_layer_size)
        self.conv3_a = nn.Conv2d(hidden_layer_size, hidden_layer_size, kernel_size=3, padding=1, groups=hidden_layer_size)
        self.conv3_b = nn.Conv2d(hidden_layer_size, hidden_layer_size//2, kernel_size=1)
        # self.residual_conv_3 = nn.Conv2d(hidden_layer_size, hidden_layer_size//2, kernel_size=1)

        self.groupnorm4 = nn.GroupNorm(32, hidden_layer_size//2)
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4_a = nn.Conv2d(hidden_layer_size//2, hidden_layer_size//2, kernel_size=3, padding=1, groups=hidden_layer_size//2)
        self.conv4_b = nn.Conv2d(hidden_layer_size//2, last_hidden_layer_size, kernel_size=1)
        # self.residual_conv_4 = nn.Conv2d(hidden_layer_size//2, last_hidden_layer_size, kernel_size=1)

        self.groupnorm5 = nn.GroupNorm(32, last_hidden_layer_size)
        self.conv5_a = nn.Conv2d(last_hidden_layer_size, last_hidden_layer_size, kernel_size=3, padding=1, groups=last_hidden_layer_size)
        self.conv5_b = nn.Conv2d(last_hidden_layer_size, num_classes, kernel_size=1)
        # self.residual_conv_5 = nn.Conv2d(last_hidden_layer_size, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.upsample1(x)
        # residual = self.residual_conv_1(x)
        x = self.groupnorm1(x)
        x = self.conv1_a(x)
        x = self.conv1_b(x)
        # x += residual

        x = self.upsample2(x)
        # residual = self.residual_conv_2(x)
        x = self.groupnorm2(x)
        x = self.conv2_a(x)
        x = self.conv2_b(x)
        # x += residual

        # residual = self.residual_conv_3(x)
        x = self.groupnorm3(x)
        x = self.conv3_a(x)
        x = self.conv3_b(x)
        # x += residual

        x = self.upsample4(x)
        # residual = self.residual_conv_4(x)
        x = self.groupnorm4(x)
        x = self.conv4_a(x)
        x = self.conv4_b(x)
        # x += residual

        # residual = self.residual_conv_5(x)
        x = self.groupnorm5(x)
        x = self.conv5_a(x)
        x = self.conv5_b(x)
        # x += residual
        return x


class Classifier(nn.Module):
    def __init__(self, num_classes, backbone='dinov2_s', head='linear', backbones=dino_backbones):
        super(Classifier, self).__init__()
        self.heads = {
            'linear': linear_head
        }
        self.backbones = dino_backbones
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.backbone.eval()
        self.head = self.heads[head](self.backbones[backbone]['embedding_size'], num_classes)

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.head(x)
        return x


class Segmentor(nn.Module):
    def __init__(self, num_classes, backbone='dinov2_s', head='conv', backbones=dino_backbones):
        super(Segmentor, self).__init__()
        self.heads = {
            'conv': conv_head,
            'depth_conv': depth_conv_head,
            'residual_depth_conv': residual_depth_conv_head
        }
        self.backbones = dino_backbones
        self.backbone = load('facebookresearch/dinov2', self.backbones[backbone]['name'])
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.num_classes = num_classes  # add a class for background if needed
        self.embedding_size = self.backbones[backbone]['embedding_size']
        self.patch_size = self.backbones[backbone]['patch_size']
        self.head = self.heads[head](self.embedding_size, self.num_classes)
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
