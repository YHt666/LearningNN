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
import json
import random
import numpy as np

from view_data import data, data_transforms, writer
from model import initialize_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('cat_to_name.json') as f:
    cat_to_name = json.load(f)


def test_model(model, dataloader):
    model.eval()
    size = len(dataloader.dataset)
    correct = 0
    class_probs = []
    class_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            class_probs_batch = [nn.Softmax()(x) for x in pred]
            class_probs.append(class_probs_batch)
            class_labels.append(labels)
    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_labels = torch.cat(class_labels)
    correct /= size
    return correct, test_probs, test_labels


def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    tb_truth = test_labels == class_index
    tb_probs = test_probs[:, class_index]

    """
    add_pr_curve参数(二分类问题)
    tag 某一类别标签
    tb_truth: List[True, False] 是否属于此类
    tb_probs: List[prob] 预测的属于此类的概率
    """   
    writer.add_pr_curve(cat_to_name[str((class_index+1).item())],
                        tb_truth, tb_probs, global_step)
    
    writer.close()


if __name__ == '__main__':
    """ toggle model """
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
    acc, test_probs, test_labels = test_model(model, dataloader)
    print(f'Test Accuracy: {100*acc}%\n')

    num_classes = len(cat_to_name)
    # 随机选10个类别展示
    showcase = random.sample(list(np.arange(num_classes)), 10)
    for i in showcase:
        add_pr_curve_tensorboard(i, test_probs, test_labels)
