import torch
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)

model.fc = nn.Linear(model.fc.in_features, 25)
