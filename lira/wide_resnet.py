import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dropout(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, n_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-ResNet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        stages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, stages[0], kernel_size=3, stride=1, padding=1)
        self.layer1 = self._wide_layer(wide_basic, stages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, stages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, stages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(stages[3], momentum=0.9)
        self.linear = nn.Linear(stages[3], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def _wide_layer(self, block, planes, n_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(n_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out



class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), #输入图片为3*32*32   same卷积，增加通道数,输出64*32*32
            nn.BatchNorm2d(num_features=64),   #强行将数据拉回到均值为0，方差为1的正态分布上;一方面使得数据分布一致，另一方面避免梯度消失。
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)     #输出64*16*16
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,128,3,1,1),   #输出128*16*16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)    #输出128*8*8
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128,256,3,1,1),  #输出256*8*8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2)   #输出256*4*4
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256,512,3,1,1),  #输出512*4*4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)   #输出512*2*2
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)   #输出512*1*1
        )
        self.fc = nn.Sequential(
            nn.Flatten(),    #输出512*1*1
            nn.Linear(in_features=512,out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,10),
        )
 
        self.model = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.fc
        )
 
 
    def forward(self,x):
        x = self.model(x)
        return x