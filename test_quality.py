import torch
import cv2
from models.model_resnet import ResNet, FaceQuality
import os
import argparse
import shutil
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Face Quality test')
parser.add_argument('--backbone', default='backbone_resume.pth', type=str, metavar='PATH',
                    help='path to backbone model')
parser.add_argument('--quality', default='quality_resume.pth', type=str, metavar='PATH',
                    help='path to quality model')
parser.add_argument('--file', default='', type=str, metavar=' PATH',
                    help='test file(image file or directory)')
parser.add_argument('--output', default='quality_result', type=str, metavar=' PATH',
                    help='output path')
parser.add_argument('--cpu', dest='cpu', action='store_true',
                    help='evaluate model on cpu')
parser.add_argument('--gpu', default=0, type=int,
                    help='index of gpu to run')

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

def get_face_quality(backbone, quality, device, img):
    resized = cv2.resize(img, (112, 112))
    ccropped = resized[...,::-1] # BGR to RGB
    # load numpy to tensor
    ccropped = ccropped.swapaxes(1, 2).swapaxes(0, 1)
    ccropped = np.reshape(ccropped, [1, 3, 112, 112])
    ccropped = np.array(ccropped, dtype = np.float32)
    ccropped = (ccropped - 127.5) / 128.0
    ccropped = torch.from_numpy(ccropped)

    # extract features
    backbone.eval() # set to evaluation mode
    with torch.no_grad():
        _, fc = backbone(ccropped.to(device), True)
        s = quality(fc)[0]

    return s.cpu().numpy()

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BACKBONE = ResNet(num_layers=100, feature_dim=512)
    QUALITY = FaceQuality(512 * 7 * 7)

    if os.path.isfile(args.backbone):
        print("Loading Backbone Checkpoint '{}'".format(args.backbone))
        checkpoint = torch.load(args.backbone, map_location='cpu')
        load_state_dict(BACKBONE, checkpoint)
    else:
        print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format(args.backbone))
        return
    if os.path.isfile(args.quality):
        print("Loading Quality Checkpoint '{}'".format(args.quality))
        checkpoint = torch.load(args.quality, map_location='cpu')
        load_state_dict(QUALITY, checkpoint)
    else:
        print("No Checkpoint Found at '{}' Please Have a Check or Continue to Train from Scratch".format(args.quality))
        return
    BACKBONE.to(DEVICE)
    QUALITY.to(DEVICE)
    BACKBONE.eval()
    QUALITY.eval()

    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output)

    if os.path.isfile(args.file):
        image = cv2.imread(args.file)
        if image is None or image.shape[0] == 0:
            print("Open image failed: ", args.file)
            return
        quality = get_face_quality(BACKBONE, QUALITY, DEVICE, image)
        cv2.imwrite('{}/{:.4f}.jpg'.format(args.output, quality[0]), image)
    elif os.path.isdir(args.file):
        for tmp in os.listdir(args.file):
            image = cv2.imread(os.path.join(args.file, tmp))
            if image is None or image.shape[0] == 0:
                print("Open image failed: ", args.file)
                continue
            quality = get_face_quality(BACKBONE, QUALITY, DEVICE, image)
            print(quality)
            cv2.imwrite('{}/{:.4f}.jpg'.format(args.output, quality[0]), image)
    else:
        print(args.file, "not exists")
        return

if __name__ == '__main__':
    main(parser.parse_args())
