import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_model(num_attrs=4):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Replace final classification layer
    model.fc = nn.Linear(model.fc.in_features, num_attrs)

    return model
