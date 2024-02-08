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

from view_data import data, data_transforms
from model import initialize_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Classifier():
    """ 将训好的模型部署为图片分类器进行单张图片预测 """
    def __init__(self, model) -> None:
        self.model = model.eval()
        self.img_tf = data_transforms['valid']

    @staticmethod
    def get_img(img_path):
        img = Image.open(img_path)
        # img = np.array(img).astype(np.int8)
        return img
    
    @staticmethod
    def show_results(img, probs, labels):
        with open('cat_to_name.json') as f:
            cat_to_name = json.load(f)
        names = [cat_to_name[str(x)] for x in labels]
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=100)
        ax[0].axis('off')
        ax[0].imshow(img)
        ax[1].barh(np.arange(5)[::-1], probs*100, tick_label=names)
        ax[1].set_xlabel('Probability (%)')
        fig.tight_layout()
        plt.show()

    def predict(self, img_path):
        with torch.no_grad():
            img = self.get_img(img_path)
            img_tensor = self.img_tf(img)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device)
            pred = nn.Softmax()(self.model(img_tensor).squeeze())
            value, index = pred.topk(5)     # 列出概率最大的5个类别
            index += 1  # 数据集的类别标签是从1开始的
        self.show_results(img, value.cpu().numpy(), index.cpu().numpy())


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

    flower_classifier = Classifier(model)
    img_path = r'images\rose.jpg'
    # img_path = r'images\sunflower.jpg'
    # img_path = r'images\lotus.jpg'
    flower_classifier.predict(img_path)
