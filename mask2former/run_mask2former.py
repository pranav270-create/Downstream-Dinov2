import torch

from mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from fpn_decode import BasePixelDecoder

# in_channels: channels of the input features
# mask_classification: whether to add mask classifier or not
# num_classes: number of classes
# hidden_dim: Transformer feature dimension
# num_queries: number of queries
# nheads: number of heads
# dim_feedforward: feature dimension in feedforward network
# enc_layers: number of Transformer encoder layers
# dec_layers: number of Transformer decoder layers
# pre_norm: whether to use pre-LayerNorm or not
# mask_dim: mask feature dimension
# enforce_input_project: add input project 1x1 conv even if input
# channels and hidden dim is identical

in_channels = 3
num_classes = 47
hidden_dim = 256
num_queries = 100
nheads = 8
dim_feedforward = 2048
dec_layers = 6
pre_norm = False
mask_dim = 256
enforce_input_project = False
mask_classification = True
transformer_model = MultiScaleMaskedTransformerDecoder(in_channels, num_classes, hidden_dim, num_queries, nheads, dim_feedforward, dec_layers, pre_norm, mask_dim, enforce_input_project, mask_classification=mask_classification)

# input_shape: shapes (channels and stride) of the input features. Dict[str, ShapeSpec],
# conv_dims: number of output channels for the intermediate conv layers.
# mask_dim: number of output channels for the final conv layer.
# norm (str or callable): normalization for all conv layers

input_shape = (3, 1)
conv_dim = 10
mask_dim = 20
decoder_model = BasePixelDecoder(input_shape, conv_dim, mask_dim)

img = torch.randn(1, 3, 512, 512)
out = decoder_model(img)
print(out)

