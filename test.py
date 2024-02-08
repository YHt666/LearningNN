import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # 指定GPU
os.environ['TORCH_HOME']='E:/PretrainedModel'   # 修改预训练模型下载路径

import torch
thread = 8 #设到10以下
torch.set_num_threads(int(thread))

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pickle
import numpy as np
import matplotlib.pylab as plt

from view_data import data, data_transforms
from model import initialize_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_model(model, dataloader):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    correct /= size
    return correct


if __name__ == '__main__':
    model_name = 'resnet'
    record_path = r'temp\train_progress\resnet\BS64LR0.01\record.pkl'
    # model_name = 'alexnet'
    # record_path = r'temp\train_progress\alexnet\BS64LR0.01\record.pkl'
    # model_name = 'vgg'
    # record_path = r'temp\train_progress\vgg\BS64LR0.01\record.pkl'

    with open(record_path, 'rb') as f:
        record = pickle.load(f, encoding='bytes')
    state_dict = record['best_record']['weight']

    model = initialize_model(model_name, 102, feature_extract=False, use_pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)

    dataloader = DataLoader(data['test'], batch_size=64, num_workers=2, pin_memory=True)

    print('Testing...')
    acc = test_model(model, dataloader)
    print(f'Test Accuracy: {100*acc}%\n')
