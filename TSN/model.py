import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()                                   
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
    
class resnet_flow(resnet):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        L = 7

        new_in_channels = 2 * L                                  

        old_w = self.backbone.conv1.weight.data                
        avg_w = old_w.mean(dim=1, keepdim=True)            
        new_w = avg_w.repeat(1, new_in_channels, 1, 1)   

        new_conv1 = nn.Conv2d(new_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight = nn.Parameter(new_w)
        self.backbone.conv1 = new_conv1