import torch
import os

class Config:
    MODEL_ROOT = 'output/deepcam_model'
    LOG_ROOT = 'output/deepcam_log'
    BACKBONE_RESUME_ROOT = './backbone_resume.pth'
    HEAD_RESUME_ROOT = './head_resume.pth'
    TRAIN_FILES = './dataset/face_train_ms1mv2.txt'

    RGB_MEAN = [0.5, 0.5, 0.5]
    RGB_STD = [0.5, 0.5, 0.5]
    EMBEDDING_SIZE = 512
    BATCH_SIZE = 5000
    DROP_LAST = True
    BACKBONE_LR = 0.05
    QUALITY_LR = 0.01
    NUM_EPOCH = 90
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9

    HEAD_GPUS = [0]
    BACKBONE_GPUS = [1, 2, 3]

    PRETRAINED_BACKBONE = 'pretrained_backbone_resume.pth'
    PRETRAINED_QUALITY = 'pretrained_qulity_resume.pth'

    NUM_EPOCH_WARM_UP = 1
    FIXED_BACKBONE_FEATURE = False

config = Config()
