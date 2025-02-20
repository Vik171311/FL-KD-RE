import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(10)
random.seed(10)
torch.cuda.manual_seed(10)
torch.cuda.manual_seed_all(10)
np.random.seed(10)
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock,block, num_classes=10):
        super(ResNet, self).__init__()
        self.block = block
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, self.block[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, self.block[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, self.block[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, self.block[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.fc(out)
    #     return out

    def forward_features(self, x, requires_feat):
        feat = []
        x = self.conv1(x)
        for i in range(self.block[0]):
            x = self.layer1[i](x)
            feat.append(x)
        for i in range(self.block[1]):
            x = self.layer2[i](x)
            feat.append(x)
        for i in range(self.block[2]):
            x = self.layer3[i](x)
            feat.append(x)
        for i in range(self.block[3]):
            x = self.layer4[i](x)
            feat.append(x)

        return (x, feat) if requires_feat else x
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        # if self.drop_rate:
        #     x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x, requires_feat=False):
        if requires_feat:
            x, feat = self.forward_features(x, requires_feat=True)
            x = self.forward_head(x, pre_logits=True)
            feat.append(x)
            x = self.fc(x)
            return x, feat
        else:
            x = self.forward_features(x, requires_feat=False)
            x = self.forward_head(x)
            return x
    def stage_info(self, stage):
        if self.block[0]==2:
            if stage == 0:
                index = 0
                shape = (64, 32, 32)
            elif stage == 1:
                index = 1
                shape = (64, 32, 32)
            elif stage == 2:
                index = 2
                shape = (128, 16, 16)
            elif stage == 3:
                index = 3
                shape = (128, 16, 16)
            elif stage == 4:
                index = 4
                shape = (256, 8, 8)
            elif stage == 5:
                index = 5
                shape = (256, 8, 8)
            elif stage == 6:
                index = 6
                shape = (512, 4, 4)
            elif stage == 7:
                index = 7
                shape = (512, 4, 4)
            elif stage == -1:
                index = -1
                shape = 512
            else:
                raise RuntimeError(f'Stage {stage} out of range (0-7)')
        elif self.block[0]==3:
            if stage == 0:
                index = 0
                shape = (64, 32, 32)
            elif stage == 1:
                index = 1
                shape = (64, 32, 32)
            elif stage == 2:
                index = 2
                shape = (64, 32, 32)
            elif stage == 3:
                index = 3
                shape = (128, 16, 16)
            elif stage == 4:
                index = 4
                shape = (128, 16, 16)
            elif stage == 5:
                index = 5
                shape = (128, 16, 16)
            elif stage == 6:
                index = 6
                shape = (128, 16, 16)
            elif stage == 7:
                index = 7
                shape = (256, 8, 8)
            elif stage == 8:
                index = 8
                shape = (256, 8, 8)
            elif stage == 9:
                index = 9
                shape = (256, 8, 8)
            elif stage == 10:
                index = 10
                shape = (256, 8, 8)
            elif stage == 11:
                index = 11
                shape = (256, 8, 8)
            elif stage == 12:
                index = 12
                shape = (256, 8, 8)
            elif stage == 13:
                index = 13
                shape = (512, 4, 4)
            elif stage == 14:
                index = 14
                shape = (512, 4, 4)
            elif stage == 15:
                index = 15
                shape = (512, 4, 4)
            elif stage == -1:
                index = -1
                shape = 512
            else:
                raise RuntimeError(f'Stage {stage} out of range (0-15)')
        return index, shape


class ResNet2(nn.Module):
    def __init__(self, ResidualBlock,block, num_classes=100):
        super(ResNet2, self).__init__()
        self.block = block
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, self.block[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, self.block[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, self.block[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, self.block[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    # def forward(self, x):
    #     out = self.conv1(x)
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     out = out.view(out.size(0), -1)
    #     out = self.fc(out)
    #     return out

    def forward_features(self, x, requires_feat):
        feat = []
        x = self.conv1(x)
        for i in range(self.block[0]):
            x = self.layer1[i](x)
            feat.append(x)
        for i in range(self.block[1]):
            x = self.layer2[i](x)
            feat.append(x)
        for i in range(self.block[2]):
            x = self.layer3[i](x)
            feat.append(x)
        for i in range(self.block[3]):
            x = self.layer4[i](x)
            feat.append(x)

        return (x, feat) if requires_feat else x
    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        # if self.drop_rate:
        #     x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x, requires_feat=False):
        if requires_feat:
            x, feat = self.forward_features(x, requires_feat=True)
            x = self.forward_head(x, pre_logits=True)
            feat.append(x)
            x = self.fc(x)
            return x, feat
        else:
            x = self.forward_features(x, requires_feat=False)
            x = self.forward_head(x)
            return x
    def stage_info(self, stage):
        if self.block[0]==2:
            if stage == 0:
                index = 0
                shape = (64, 32, 32)
            elif stage == 1:
                index = 1
                shape = (64, 32, 32)
            elif stage == 2:
                index = 2
                shape = (128, 16, 16)
            elif stage == 3:
                index = 3
                shape = (128, 16, 16)
            elif stage == 4:
                index = 4
                shape = (256, 8, 8)
            elif stage == 5:
                index = 5
                shape = (256, 8, 8)
            elif stage == 6:
                index = 6
                shape = (512, 4, 4)
            elif stage == 7:
                index = 7
                shape = (512, 4, 4)
            elif stage == -1:
                index = -1
                shape = 512
            else:
                raise RuntimeError(f'Stage {stage} out of range (0-7)')
        elif self.block[0]==3:
            if stage == 0:
                index = 0
                shape = (64, 32, 32)
            elif stage == 1:
                index = 1
                shape = (64, 32, 32)
            elif stage == 2:
                index = 2
                shape = (64, 32, 32)
            elif stage == 3:
                index = 3
                shape = (128, 16, 16)
            elif stage == 4:
                index = 4
                shape = (128, 16, 16)
            elif stage == 5:
                index = 5
                shape = (128, 16, 16)
            elif stage == 6:
                index = 6
                shape = (128, 16, 16)
            elif stage == 7:
                index = 7
                shape = (256, 8, 8)
            elif stage == 8:
                index = 8
                shape = (256, 8, 8)
            elif stage == 9:
                index = 9
                shape = (256, 8, 8)
            elif stage == 10:
                index = 10
                shape = (256, 8, 8)
            elif stage == 11:
                index = 11
                shape = (256, 8, 8)
            elif stage == 12:
                index = 12
                shape = (256, 8, 8)
            elif stage == 13:
                index = 13
                shape = (512, 4, 4)
            elif stage == 14:
                index = 14
                shape = (512, 4, 4)
            elif stage == 15:
                index = 15
                shape = (512, 4, 4)
            elif stage == -1:
                index = -1
                shape = 512
            else:
                raise RuntimeError(f'Stage {stage} out of range (0-15)')
        return index, shape

def ResNet18():
    return ResNet(ResidualBlock,[2,2,2,2],num_classes=100)
def ResNet34():
    return ResNet(ResidualBlock,[3,4,6,3],num_classes=10)
def ResNet34_2():
    return ResNet(ResidualBlock,[3,4,6,3],num_classes=10)
def ResNet34_3():
    return ResNet2(ResidualBlock,[3,4,6,3],num_classes=100)
def ResNet34_4():
    return ResNet2(ResidualBlock,[3,4,6,3],num_classes=100)