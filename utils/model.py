import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import swin_v2_b, resnet152
from torchvision.models import Swin_V2_B_Weights, ResNet152_Weights
import timm
from utils.sam import SAM

class Detector(nn.Module):
    
    def __init__(self, model="hrnet_w18_multi", lr=5e-4):
        super(Detector, self).__init__()
        if model == "efficientnet-b4":
            self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        elif model == "efficientnet-b7":
            self.net=EfficientNet.from_pretrained("efficientnet-b7",advprop=True,num_classes=2)
        elif model == "swin":
            self.net=swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            self.net.head = nn.Linear(in_features=1024, out_features=2, bias=True)
        elif model == "resnet":
            self.net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            self.net.head = nn.Linear(in_features=1024, out_features=2, bias=True)
        elif model == "hrnet_w18":
            self.net = timm.create_model('hrnet_w18', pretrained=True, num_classes=2)
        elif model == "hrnet_w32":
            self.net = timm.create_model('hrnet_w32', pretrained=True, num_classes=2)
        elif model == "hrnet_w44":
            self.net = timm.create_model('hrnet_w44', pretrained=True, num_classes=2)
        elif model == "hrnet_w64":
            self.net = timm.create_model('hrnet_w64', pretrained=True, num_classes=2)
            
        self.cel=nn.CrossEntropyLoss()
        self.optimizer=SAM(self.parameters(),torch.optim.SGD,lr=lr, momentum=0.9)

    def forward(self,x):
        x=self.net(x)
        return x
    
    def training_step(self,x,target):
        for i in range(2):
            pred_cls=self(x)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            loss=loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)
        
        return pred_first
    