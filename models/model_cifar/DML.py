import torch
import torch.nn as nn
from .resnet import *
from .densenet import *
from .vgg import *

__all__ = ['MutualNet']


class MutualNet(nn.Module):
    def __init__(self, model="resnet32", num_branches=4, num_classes=10, dropout=0.0):
        super().__init__()
        self.num_branches = num_branches

        for i in range(num_branches):
            if model == "resnet32":
                # setattr用来对对象加入属性，建立DML的多个分支
                setattr(self, 'stu' + str(i), resnet32(num_classes=num_classes))
            elif model == "resnet110":
                setattr(self, 'stu' + str(i), resnet110(num_classes=num_classes))
            elif model == "vgg16":
                setattr(self, 'stu' + str(i), vgg16(num_classes=num_classes, dropout=dropout))
            elif model == "densenetd40k12":
                setattr(self, 'stu' + str(i), densenetd40k12(num_classes=num_classes))
            elif model == "wide_resnet20_8":
                setattr(self, 'stu' + str(i), wide_resnet20_8(num_classes=num_classes))

    def forward(self, x):
        out = self.stu0(x)
        out = out.unsqueeze(-1)
        for i in range(1, self.num_branches):
            temp_out = getattr(self, 'stu' + str(i))(x)
            temp_out = temp_out.unsqueeze(-1)  # 在末尾增加一个维度，表示网络分支，将所有网络分支的prediction拼接在一起求loss
            out = torch.cat([out, temp_out], -1)
        return out
