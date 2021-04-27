import torch
import torch.nn as nn
import torch.optim as optim
import math
import torchvision.transforms as transforms
import cv2
from dataset.dataset import ImageFolder
from config import config
from models.model_resnet import ResNet, FaceQuality
from models.metrics import GaussianFace
from models.focal import FocalLoss
from util.utils import *
from util.cosine_lr_scheduler import CosineDecayLR
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import random
import numbers
import shutil
import argparse
import numpy as np
from ptflops import get_model_complexity_info

def load_state_dict(model, state_dict):
    all_keys = {k for k in state_dict.keys()}
    for k in all_keys:
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    if len(pretrained_dict) == len(model_dict):
        print("all params loaded")
    else:
        not_loaded_keys = {k for k in pretrained_dict.keys() if k not in model_dict.keys()}
        print("not loaded keys:", not_loaded_keys)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

def train():
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(config.LOG_ROOT)

    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomResizedCrop(112, scale=(0.95, 1), ratio=(1, 1))]),
        transforms.Resize(112),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(0.01),
        transforms.ToTensor(),
        transforms.Normalize(mean = config.RGB_MEAN, std = config.RGB_STD),
    ])

    dataset_train = ImageFolder(config.TRAIN_FILES, train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = config.BATCH_SIZE, pin_memory = True, shuffle=True,
        num_workers = 8, drop_last = True
    )

    NUM_CLASS = train_loader.dataset.classes
    print("Number of Training Classes: {}".format(NUM_CLASS))

    BACKBONE = ResNet(num_layers=100, feature_dim=512)
    flops, params = get_model_complexity_info(BACKBONE, (3, 112, 112), as_strings=True, print_per_layer_stat=False)
    print('BACKBONE FLOPs:', flops)
    print('BACKBONE PARAMS:', params)

    PRETRAINED_BACKBONE = None
    PRETRAINED_QUALITY = None

    if os.path.isfile(config.PRETRAINED_BACKBONE) and os.path.isfile(config.PRETRAINED_QUALITY):
        PRETRAINED_BACKBONE = ResNet(num_layers=100, feature_dim=512)
        PRETRAINED_QUALITY = FaceQuality(512 * 7 * 7)
        checkpoint = torch.load(config.PRETRAINED_BACKBONE)
        load_state_dict(PRETRAINED_BACKBONE, checkpoint)
        PRETRAINED_BACKBONE = nn.DataParallel(PRETRAINED_BACKBONE, device_ids = config.BACKBONE_GPUS)
        PRETRAINED_BACKBONE = PRETRAINED_BACKBONE.cuda(0)
        PRETRAINED_BACKBONE.eval()

        checkpoint = torch.load(config.PRETRAINED_QUALITY)
        load_state_dict(PRETRAINED_QUALITY, checkpoint)
        PRETRAINED_QUALITY = nn.DataParallel(PRETRAINED_QUALITY, device_ids = config.BACKBONE_GPUS)
        PRETRAINED_QUALITY = PRETRAINED_QUALITY.cuda(0)
        PRETRAINED_QUALITY.eval()

    HEAD = GaussianFace(in_features = config.EMBEDDING_SIZE, out_features = NUM_CLASS)
    LOSS = FocalLoss()
    # optionally resume from a checkpoint
    if config.BACKBONE_RESUME_ROOT and config.HEAD_RESUME_ROOT:
        print("=" * 60)
        if os.path.isfile(config.BACKBONE_RESUME_ROOT):
            print("Loading Backbone Checkpoint '{}'".format(config.BACKBONE_RESUME_ROOT))
            checkpoint = torch.load(config.BACKBONE_RESUME_ROOT)
            load_state_dict(BACKBONE, checkpoint)
        else:
            print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format(config.BACKBONE_RESUME_ROOT))
        if os.path.isfile(config.HEAD_RESUME_ROOT):
            print("Loading Head Checkpoint '{}'".format(config.HEAD_RESUME_ROOT))
            checkpoint = torch.load(config.HEAD_RESUME_ROOT)
            load_state_dict(HEAD, checkpoint)
        else:
            print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format(config.HEAD_RESUME_ROOT))
        print("=" * 60)

    BACKBONE = nn.DataParallel(BACKBONE, device_ids = config.BACKBONE_GPUS, output_device=config.BACKBONE_GPUS[-1])
    BACKBONE = BACKBONE.cuda(config.BACKBONE_GPUS[0])
    HEAD = nn.DataParallel(HEAD, device_ids = config.HEAD_GPUS, output_device=config.HEAD_GPUS[0])
    HEAD = HEAD.cuda(config.HEAD_GPUS[0])
    OPTIMIZER = optim.SGD([
            {'params': BACKBONE.parameters(), 'lr': config.BACKBONE_LR, 'weight_decay': config.WEIGHT_DECAY},
            {'params': HEAD.parameters(), 'lr': config.BACKBONE_LR}
        ],
        momentum=config.MOMENTUM)
    DISP_FREQ = len(train_loader) // 100

    NUM_EPOCH_WARM_UP = config.NUM_EPOCH_WARM_UP
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP
    batch = 0
    step = 0

    scheduler = CosineDecayLR(OPTIMIZER, T_max = 10*len(train_loader), lr_init = config.BACKBONE_LR, lr_min = 1e-5, warmup = NUM_BATCH_WARM_UP)
    for epoch in range(config.NUM_EPOCH):
        BACKBONE.train()
        HEAD.train()
        arcface_losses = AverageMeter()
        confidences = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        scaler = torch.cuda.amp.GradScaler()
        for inputs, labels in tqdm(iter(train_loader)):
            inputs = inputs.cuda(config.BACKBONE_GPUS[0])
            labels = labels.cuda(config.HEAD_GPUS[0])
            with torch.cuda.amp.autocast():
                features = BACKBONE(inputs)
                if PRETRAINED_BACKBONE is None or PRETRAINED_QUALITY is None:
                    outputs = HEAD(None, features.cuda(config.HEAD_GPUS[0]), labels, False)
                else:
                    with torch.no_grad():
                        _, fc = PRETRAINED_BACKBONE(inputs, True)
                        quality = PRETRAINED_QUALITY(fc)
                    outputs = HEAD(quality.cuda(config.HEAD_GPUS[0]), features.cuda(config.HEAD_GPUS[0]), labels, True)
            # measure accuracy and record loss
            arcface_loss = LOSS(outputs, labels)
            prec1, prec5 = accuracy(outputs.data, labels, topk = (1, 5))
            arcface_losses.update(arcface_loss.data.item(), inputs.size(0))
            top1.update(prec1.data.item(), inputs.size(0))
            top5.update(prec5.data.item(), inputs.size(0))
            loss = arcface_loss
            # compute gradient and do SGD step
            OPTIMIZER.zero_grad()
            #loss.backward()
            #OPTIMIZER.step()
            scaler.scale(loss).backward()
            scaler.step(OPTIMIZER)
            scaler.update()
            if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                print("=" * 60)
                print('Epoch {}/{} Batch {}/{}\t'
                      'Training Loss {arcface_loss.val:.4f} ({arcface_loss.avg:.4f})\t'
                      'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch + 1, config.NUM_EPOCH, batch + 1, len(train_loader) * config.NUM_EPOCH,
                    arcface_loss = arcface_losses,  top1 = top1, top5 = top5))
                print("=" * 60)

            batch += 1 # batch index
            scheduler.step(batch)
            if batch % 1000 == 0:
                print(OPTIMIZER)
        # training statistics per epoch (buffer for visualization)
        epoch_loss = arcface_losses.avg
        epoch_acc = top1.avg
        writer.add_scalar("Training_Loss", epoch_loss, epoch + 1)
        writer.add_scalar("Training_Accuracy", epoch_acc, epoch + 1)
        print("=" * 60)
        print('Epoch: {}/{}\t'
              'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            epoch + 1, config.NUM_EPOCH, loss = arcface_losses, top1 = top1, top5 = top5))
        print("=" * 60)

        # save checkpoints per epoch
        curTime = get_time()
        if not os.path.exists(config.MODEL_ROOT):
            os.makedirs(config.MODEL_ROOT)
        torch.save(BACKBONE.state_dict(), os.path.join(config.MODEL_ROOT, "Backbone_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(epoch + 1, batch, curTime)))
        torch.save(HEAD.state_dict(), os.path.join(config.MODEL_ROOT, "Head_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(epoch + 1, batch, curTime)))

if __name__ == "__main__":
    train()
