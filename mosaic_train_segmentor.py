# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Example script to train a Dinov2 + Segmentation Head model for semantic segmentation."""

"""TODO:
1) ViT Implementation
3) MaskFormer
4) More augmentations?
"""

import argparse
import logging
import os

import torch
import torchvision
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from composer import DataSpec, Time, Trainer
from composer.algorithms import EMA, SAM, ChannelsLast, MixUp, ProgressiveResizing, LayerFreezing, LowPrecisionGroupNorm, LowPrecisionLayerNorm
from composer.callbacks import CheckpointSaver, ImageVisualizer, LRMonitor, SpeedMonitor
# from composer.datasets.ade20k import (ADE20k, PadToSize, PhotometricDistoration, RandomCropPair, RandomHFlipPair,
#                                       RandomResizePair)
# from composer.datasets.utils import NormalizationFn, pil_image_collate
from composer.loggers import WandBLogger
from composer.loss import DiceLoss, soft_cross_entropy
from composer.metrics import CrossEntropy, MIoU
from composer.models import ComposerClassifier
# from composer.models.deeplabv3.model import deeplabv3
from composer.optim import CosineAnnealingScheduler, DecoupledSGDW, DecoupledAdamW
from composer.utils import dist

from sklearn.model_selection import train_test_split

from model import Segmentor
from dataset import SegmentationDataset

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()

# Dataloader command-line arguments
parser.add_argument('--data_dir', help='Path to the directory containing the ImageNet-1k dataset', type=str, default="data")
parser.add_argument('--download',
                    help='Use to download ADE20k from the internet and put it in the `data_dir`',
                    action='store_true')
parser.add_argument('--train_resize_size', help='Training image resize size', type=int, default=896)
parser.add_argument('--eval_resize_size', help='Evaluation image resize size', type=int, default=896)
parser.add_argument('--train_batch_size', help='Train dataloader per-device batch size', type=int, default=1)
parser.add_argument('--eval_batch_size', help='Validation dataloader per-device batch size', type=int, default=1)

# Model command-line arguments
parser.add_argument('--backbone_arch',
                    help='Architecture to use for the backbone.',
                    default='resnet101',
                    choices=['resnet50', 'resnet101'])
parser.add_argument('--sync_bn',
                    help='Use sync BatchNorm. Recommended if the per device microbatch size is below 16',
                    action='store_true')
parser.add_argument('--cross_entropy_weight', help='Weight to scale the cross entropy loss', type=float, default=0.375)
parser.add_argument('--dice_weight', help='Weight to scale the dice loss', type=float, default=1.125)
parser.add_argument('--focal_weight', help='Weight to scale the focal loss', type=float, default=1.5)

# Optimizer command-line arguments
parser.add_argument('--learning_rate', help='Optimizer learning rate', type=float, default=1e-4)
parser.add_argument('--momentum', help='Optimizer momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', help='Optimizer weight decay', type=float, default=5.0e-5)

# Save checkpoint command-line arguments
parser.add_argument('--save_checkpoint_dir',
                    help='Directory in which to save model checkpoints',
                    type=str,
                    default='checkpoints/{run_name}')
parser.add_argument('--checkpoint_interval',
                    help='Frequency to save checkpoints',
                    type=Time.from_timestring,
                    default='1ep')

# Load checkpoint command-line arguments, assumes resuming from a previous training run (as opposed to fine-tuning)
parser.add_argument('--load_checkpoint_path', help='Path to the checkpoint to load', type=str)

# Recipes command-line argument
parser.add_argument('--recipe_name',
                    help='Algorithmic recipes to be applied to the trainer',
                    choices=['mild', 'medium', 'hot'])

# Logger command-line arguments
# Note: Only Weights and Biases to minimize arguments. Other loggers can be used by adjusting the script
parser.add_argument('--wandb_logger', help='Whether or not to log results to Weights and Biases', action='store_true')
parser.add_argument('--wandb_entity', help='WandB entity name', type=str)
parser.add_argument('--wandb_project', help='WandB project name', type=str)

parser.add_argument('--image_viz', help='Whether or not to log images using ImageVisualizer', action='store_true')

# Trainer arguments
parser.add_argument('--grad_accum', help='Amount of gradient accumulation if running on CPU', type=int, default=1)
parser.add_argument('--run_name', help='Name of the training run used for checkpointing and logging', type=str)
parser.add_argument('--seed', help='Random seed', type=int, default=42)
parser.add_argument('--max_duration',
                    help='Duration to train specified as a Time string',
                    type=Time.from_timestring,
                    default='128ep')

args = parser.parse_args()

IMAGENET_CHANNEL_MEAN = (int(0.485 * 255), int(0.456 * 255), int(0.406 * 255))
IMAGENET_CHANNEL_STD = (int(0.229 * 255), int(0.224 * 255), int(0.225 * 255))

# ADE20K_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
# ADE20K_FILE = 'ADEChallengeData2016.zip'


def _main():
    # Divide batch size by number of devices
    if dist.get_world_size() > 1:
        args.train_batch_size = args.train_batch_size // dist.get_world_size()
        args.eval_batch_size = args.eval_batch_size // dist.get_world_size()

    # Train dataset code
    logging.info('Building train dataloader')

    if False:
        torchvision.datasets.utils.download_and_extract_archive(url=ADE20K_URL,
                                                                download_root=args.data_dir,
                                                                filename=ADE20K_FILE,
                                                                remove_finished=True)
        # Adjust the data_dir to include the extracted directory
        args.data_dir = os.path.join(args.data_dir, 'ADEChallengeData2016')

    # Training transforms applied to both the image and target
    # train_both_transforms = torch.nn.Sequential(
    #     RandomResizePair(
    #         min_scale=0.5,
    #         max_scale=2.0,
    #         base_size=(args.train_resize_size, args.train_resize_size),
    #     ),
    #     RandomCropPair(
    #         crop_size=(args.train_resize_size, args.train_resize_size),
    #         class_max_percent=0.75,
    #         num_retry=10,
    #     ),
    #     RandomHFlipPair(),
    # )

    # Training transforms applied to the image only
    # train_image_transforms = torch.nn.Sequential(
    #     PhotometricDistoration(
    #         brightness=32. / 255,
    #         contrast=0.5,
    #         saturation=0.5,
    #         hue=18. / 255,
    #     ),
    #     PadToSize(
    #         size=(args.train_resize_size, args.train_resize_size),
    #         fill=IMAGENET_CHANNEL_MEAN,
    #     ),
    # )

    # Training transforms applied to the target only
    # train_target_transforms = PadToSize(size=(args.train_resize_size, args.train_resize_size), fill=0)

    base_size = (32, 32)
    img_transform = transforms.Compose([
        transforms.Resize((14*base_size[0], 14*base_size[1]), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((8*base_size[0], 8*base_size[1]), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    num_classes = 47
    dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes=num_classes, img_transform=img_transform, mask_transform=mask_transform)
    train_imgs, valid_imgs = train_test_split(dataset.images, test_size=0.2, random_state=args.seed)
    train_dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes=num_classes, img_transform=img_transform, mask_transform=mask_transform, images=train_imgs)
    val_dataset = SegmentationDataset(img_dir="data/train", mask_dir="data/masks", num_classes=num_classes, img_transform=img_transform, mask_transform=mask_transform, images=valid_imgs)

    # Create ADE20k train dataset
    # train_dataset = ADE20k(
    #     datadir=args.data_dir,
    #     split='training',
    #     image_transforms=train_image_transforms,
    #     target_transforms=train_target_transforms,
    #     both_transforms=train_both_transforms,
    # )

    train_sampler = None
    if dist.get_world_size():
        # Nifty function to instantiate a PyTorch DistributedSampler based on your hardware setup
        train_sampler = dist.get_sampler(train_dataset, drop_last=True, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=True,  # Prevents using a smaller batch at the end of an epoch
        sampler=train_sampler,
        # collate_fn=pil_image_collate,
        persistent_workers=True,
    )

    # DataSpec enables image normalization to be performed on-GPU, marginally relieving dataloader bottleneck
    train_dataspec = DataSpec(dataloader=train_dataloader)

    logging.info('Built train dataloader\n')

    # Validation dataset code
    logging.info('Building evaluation dataloader')

    # Validation image and target transformations
    image_transforms = transforms.Resize(size=(args.eval_resize_size, args.eval_resize_size),
                                         interpolation=InterpolationMode.BILINEAR)
    target_transforms = transforms.Resize(size=(args.eval_resize_size, args.eval_resize_size),
                                          interpolation=InterpolationMode.NEAREST)

    # Create validation dataset
    # val_dataset = ADE20k(datadir=args.data_dir,
    #                      split='validation',
    #                      both_transforms=None,
    #                      image_transforms=image_transforms,
    #                      target_transforms=target_transforms)

    #Create validation dataloader

    val_sampler = None
    if dist.get_world_size():
        # Nifty function to instantiate a PyTorch DistributedSampler based on your hardware
        val_sampler = dist.get_sampler(val_dataset, drop_last=False, shuffle=False)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler,
        # collate_fn=pil_image_collate,
        persistent_workers=True,
    )

    # DataSpec enables image normalization to be performed on-GPU, marginally relieving dataloader bottleneck
    val_dataspec = DataSpec(dataloader=val_dataloader)
    logging.info('Built validation dataset\n')

    # Create a model
    model = Segmentor(num_classes, backbone='dinov2_l', head="residual_depth_conv")
    # loadpath = "checkpoints/mosaic_seg/ep2-ba18078-rank0.pt"
    # model.load_state_dict(torch.load(loadpath))
    # model = torch.compile(model)

    # Initialize the classifier head only since the backbone uses pre-trained weights
    # def weight_init(module: torch.nn.Module):
    #     if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
    #         torch.nn.init.kaiming_normal_(module.weight)
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         torch.nn.init.ones_(module.weight)
    #         torch.nn.init.zeros_(module.bias)

    # model.classifier.apply(weight_init)  # type: ignore Does not recognize classifier as a torch.nn.Module

    # Loss function to use during training
    # This ignores index -1 since the NormalizationFn transformation sets the background class to -1
    dice_loss_fn = DiceLoss(softmax=True, batch=True, ignore_absent_classes=True)

    weights = torch.ones(num_classes)
    weights[-1] = 0.01  # Reduce the weight for the background class
    weight_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nn_cross_entropy = torch.nn.CrossEntropyLoss(weight=weights.to(weight_device))
    def combo_loss(output, target):
        loss = {}
        loss['cross_entropy'] = nn_cross_entropy(output, target)
        loss['dice'] = dice_loss_fn(output, target)
        loss['focal'] = sigmoid_focal_loss(output, target, 0.25, 2.0, reduction='mean')
        loss['total'] = args.cross_entropy_weight * loss['cross_entropy'] + args.dice_weight * loss['dice'] + args.focal_weight * loss['focal']
        return loss

    # Training and Validation metrics to log throughout training
    train_metrics = MetricCollection([CrossEntropy(), MIoU(num_classes=47)])
    val_metrics = MetricCollection([CrossEntropy(), MIoU(num_classes=47)])

    # Create a ComposerClassifier using the model, loss function, and metrics
    composer_model = ComposerClassifier(module=model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=combo_loss)

    logging.info('Built Composer model\n')

    logging.info('Building optimizer and learning rate scheduler')
    # Optimizer
    # optimizer = DecoupledSGDW(composer_model.parameters(),
    #                           lr=args.learning_rate,
    #                           momentum=args.momentum,
    #                           weight_decay=args.weight_decay)

    optimizer = DecoupledAdamW(composer_model.parameters(),
                              lr=args.learning_rate,
                              weight_decay=args.weight_decay)

    # Only use a LR schedule if no recipe is specified or if the hot recipe was specified
    lr_scheduler = None
    if args.recipe_name is None or args.recipe_name == 'hot':
        lr_scheduler = CosineAnnealingScheduler()

    logging.info('Built optimizer and learning rate scheduler')

    logging.info('Building callbacks: SpeedMonitor, LRMonitor, and CheckpointSaver')
    speed_monitor = SpeedMonitor(window_size=50)  # Measures throughput as samples/sec and tracks total training time
    lr_monitor = LRMonitor()  # Logs the learning rate

    # Callback for checkpointing
    checkpoint_saver = CheckpointSaver(folder=args.save_checkpoint_dir, save_interval=args.checkpoint_interval)
    logging.info('Built callbacks: SpeedMonitor, LRMonitor, and CheckpointSaver\n')

    # Recipes for training DeepLabv3+ on ImageNet in order of increasing training time and accuracy
    # To learn about individual methods, check out "Methods Overview" in our documentation: https://docs.mosaicml.com/
    progressive_resizing_algorithm = ProgressiveResizing(
                                mode='resize',
                                initial_scale=0.5,
                                finetune_fraction=0.5,
                                delay_fraction=0.2,
                                size_increment=4,
                                resize_targets=True  # important for semantic segmentation here
                            )
    # layer_freezing_algorithm = LayerFreezing(freeze_start=0.1, freeze_level=1.0)
    logging.info('Building algorithm recipes')
    if args.recipe_name == 'mild':
        algorithms = [
            ChannelsLast(),
            EMA(half_life='1000ba', update_interval='10ba'),
        ]
    elif args.recipe_name == 'medium':
        algorithms = [
            ChannelsLast(),
            EMA(half_life='1000ba', update_interval='10ba'),
            SAM(rho=0.3, interval=2),
            MixUp(alpha=0.2),
            progressive_resizing_algorithm
        ]
    elif args.recipe_name == 'hot':
        algorithms = [
            ChannelsLast(),
            EMA(half_life='2000ba', update_interval='1ba'),
            SAM(rho=0.3, interval=1),
            MixUp(alpha=0.5),
            progressive_resizing_algorithm
        ]
    else:
        algorithms = None

    logging.info('Built algorithm recipes\n')

    # Weight and Biases logger if specified in commandline
    logger = None
    if args.wandb_logger:
        logging.info('Building Weights and Biases logger')
        if args.wandb_entity is None:
            raise ValueError('Please specify --wandb_entity argument')
        if args.wandb_project is None:
            raise ValueError('Please specify --wandb_project argument')
        logger = WandBLogger(entity=args.wandb_entity, project=args.wandb_project)
        logging.info('Built Weights and Biases logger')

    callbacks = [speed_monitor, lr_monitor, checkpoint_saver]
    if args.image_viz:
        callbacks.append(ImageVisualizer(mode='segmentation'))
    # Create the Trainer!
    logging.info('Building Trainer')
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    precision = 'amp_fp16' if device == 'gpu' else 'fp32'  # Mixed precision for fast training when using a GPU
    grad_accum = 'auto' if device == 'gpu' else args.grad_accum  # If on GPU, use 'auto' gradient accumulation
    trainer = Trainer(run_name=args.run_name,
                      model=composer_model,
                      train_dataloader=train_dataspec,
                      eval_dataloader=val_dataspec,
                      eval_interval='1ep',
                      optimizers=optimizer,
                      schedulers=lr_scheduler,
                      algorithms=algorithms,
                      loggers=logger,
                      max_duration=args.max_duration,
                      callbacks=callbacks,
                    #   load_path=args.load_checkpoint_path,
                      device=device,
                      device_train_microbatch_size=4,
                      precision=precision,
                    #   load_weights_only=True,
                      seed=args.seed)
    logging.info('Built Trainer\n')

    # Start training!
    logging.info('Train!')
    trainer.fit()


if __name__ == '__main__':
    _main()
