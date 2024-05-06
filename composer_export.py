from composer.utils import export_for_inference
from composer.models import ComposerClassifier
from composer import DataSpec, Time, Trainer

import os
import torch

from model import Segmentor

checkpoint_path = os.path.join('checkpoints', 'mosaic_seg_d_conv_lg', 'latest-rank0.pt')

model = Segmentor(47, backbone='dinov2_l', head='depth_conv')
model = ComposerClassifier(module=model)
trainer = Trainer(model=model, load_path=checkpoint_path, load_weights_only=True)
module = trainer.state.model.module
state_dict = model.state_dict()
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
torch.save(new_state_dict, 'weights/mosaic_seg_d_conv_lg.pt')
