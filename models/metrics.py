from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class TeacherLoss(nn.Module):
    def __init__(self):
        super(TeacherLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, confidence, feature, teacher_feature, gaussian=False):
        loss = self.mse(F.normalize(feature), F.normalize(teacher_feature))
        if gaussian:
            loss = loss * confidence
        loss = loss.sum() / feature.size(0)
        return loss

class GaussianFace(nn.Module):
    def __init__(self, in_features, out_features, s = 64, m = 0.5):
        super(GaussianFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, confidence, input, label, gaussian=True):
        weight = F.normalize(self.weight)
        cosine = F.linear(F.normalize(input), weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = phi.half()
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = torch.where(one_hot==0, cosine, phi)
        if gaussian:
            confidence = torch.clamp(confidence - 0.2, 0, 1) * 1.2
            output = output * self.s * confidence
        else:
            output = output * self.s
        return output
