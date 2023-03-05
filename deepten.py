import torch
import torch.nn as nn

from encoding.nn import Encoding, View, Normalize
from encoding.models.backbone import resnet50s

path = '/home/tangb_lab/cse30011373/jiays/classifier/model/'

__all__ = ['DeepTen', 'DeepTen2']

class DeepTen(nn.Module):
    def __init__(self, nclass):
        super(DeepTen, self).__init__()

        # copying modules from pretrained models
        self.pretrained = resnet50s(pretrained=True,dilated=False)
        
        n_codes = 32
        self.head = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128,K=n_codes),
            View(-1, 128*n_codes),
            Normalize(),
            nn.Linear(128*n_codes, nclass),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        return self.head(x)

class DeepTen2(nn.Module):
    def __init__(self, nclass):
        super(DeepTen2, self).__init__()

        # copying modules from pretrained models
        self.pretrained = resnet50s(pretrained=True,root=path,dilated=False)

        n_codes = 32
        self.head = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoding(D=128,K=n_codes),
            View(-1, 128*n_codes),
            Normalize(),
        )

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        x = self.pretrained.layer4(x)
        return self.head(x)
