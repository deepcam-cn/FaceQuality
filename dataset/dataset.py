from torch.utils import data
from PIL import Image
import random
import os
import os.path
import sys

import cv2
import numpy as np

def random_compress(img):
    rand_num = random.randint(40, 90)
    img_encode = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY),rand_num])
    data_encode = np.array(img_encode[1])
    str_encode = data_encode.tostring()
    nparr = np.fromstring(str_encode, np.uint8)
    img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_decode

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def cv2_loader(path):
    img1 = cv2.imread(path)
    if np.random.random() < 0.5:
        size = np.random.choice([60, 80, 100])
        img1 = cv2.resize(img1, (size, size))
    img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img2)
    return img

class ImageFolder(data.Dataset):
    def __init__(self, trainList, transform=None, loader=None):
        super(ImageFolder, self).__init__()
        self.transform = transform
        if loader is None:
            self.loader = cv2_loader
        else:
            self.loader = loader
        with open(trainList) as f:
            self.samples = f.readlines()
        self.classes = int(self.samples[-1].split(';')[1]) + 1

    def __getitem__(self, index):
        path, target = self.samples[index].split(';')
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

