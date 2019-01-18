import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math
from torchvision import models


###################### Network Selection ######################
def NetworkSelect(opt):
    if opt.arch=='resnet': return ResNet50(opt)




###################### ResNet50 ######################
class ResNet50(nn.Module):
    def __init__(self, opt):
        self.pars = opt

        super(ResNet50, self).__init__()

        if opt.pretrained: print('Getting pretrained weights...')
        self.feature_extraction = models.resnet50(pretrained=False)
        self.feature_extraction.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extraction = nn.Sequential(*list(self.feature_extraction)[:-1])
        self.merge_tiles_2_pred = nn.Sequential(nn.Linear(opt.num_tiles*self.feature_extraction.fc.in_features, opt.num_classes), nn.Softmax(dim=1))

         self.__initialize_weights()

    def __initialize_weights(self):
        for idx,module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0,0.01)
                module.bias.data.zero_()

    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.feature_extraction(x[i])
            x_list.append(z)

        x = torch.cat(x_list, dim=1)

        x = self.merge_tiles_2_pred(x)

        return x
