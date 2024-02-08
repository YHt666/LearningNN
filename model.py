import os
os.environ['TORCH_HOME']='E:/PretrainedModel'

from torch import nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad_(False)


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    if model_name == 'resnet':
        if use_pretrained:
            model_ft = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        else:
            model_ft = models.resnet152()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'alexnet':
        if use_pretrained:
            model_ft = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        else:
            model_ft = models.alexnet()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg':
        if use_pretrained:
            model_ft = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        else:
            model_ft = models.vgg16()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    else:
        print('Invalid model name, exiting...')
        exit()
    return model_ft