import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from torchvision import transforms

import torch.nn as nn
import torch

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 128))
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=3)
        self.fc_out1 = nn.Linear(256, 128)
        self.fc_out2 = nn.Linear(128, num_classes)
       
    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :])
            out, hidden = self.lstm(x.unsqueeze(0), hidden)
        
        x = self.fc_out1(out[-1, : , :])
        x = F.relu(x)
        x = self.fc_out2(x)

        return x

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

if __name__ == "__main__":
    img = torch.randn((2, 10, 3, 224, 224)).cuda()
    model = CNNLSTM().cuda()
    print(count_parameters(model))


    out = model(img)
    print(out.shape)