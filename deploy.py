import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # 指定GPU
os.environ['TORCH_HOME']='E:/PretrainedModel'   # 修改预训练模型下载路径

import torch
thread = 8 #设到10以下
torch.set_num_threads(int(thread))

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
import pickle
import numpy as np
import matplotlib.pylab as plt
import json

from view_data import data, data_transforms, writer
from model import initialize_model
from test import cat_to_name


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


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

    def analyze(self, img_path):
        """ 可视化特征图和卷积核，对CNN有一个更深入的洞察 """
        model = self.model
        model = model.cpu()

        img = self.get_img(img_path)
        img_tensor = self.img_tf(img)
        writer.add_image('0-img', img_tensor) # 输出原图
        img_tensor = img_tensor.unsqueeze(0)

        model_name = model._get_name()
        feature_map = []
        kernel = []
        f = img_tensor
        
        if model_name == 'ResNet':
            name_module = [(name, module) for name, module in model.named_children()]
            names, modules = zip(*name_module)
            print(names)
            # 忽略全连接层
            names = names[:-1]
            modules = modules[:-1]
            
            for m in modules:
                f = m(f)
                feature_map.append(f.permute(1, 0, 2, 3))
            
            # 输出所有层的前n个 feature map
            n = 25
            r = 5   # 输出行数
            for id, (name, fm) in enumerate(zip(names, feature_map)):
                fm_grid = utils.make_grid(fm[:n], r)
                writer.add_image(f'[{id}]{name}/feature_map', fm_grid)
                # print(name)

            # 输出第一层的kernel
            kn1 = modules[0].weight
            kn_grid = utils.make_grid(kn1[:n], r)
            writer.add_image(f'[0]{names[0]}/kernel', kn_grid)

        elif model_name == 'AlexNet':
            name_module = [(name, module) for name, module in model.features.named_children()]
            names, modules = zip(*name_module)
            print(names)
            print(model.features)

            # 输出各卷积层的 feature map 和 kernel
            for name, module in name_module:
                f = module(f)
                if name in ['0', '3', '6', '8', '10']:
                    feature_map.append(f.permute(1, 0, 2, 3))
                    kernel.append(module.weight)

            n = 25
            r = 5   # 输出行数
            for id, (name, fm, kn) in enumerate(zip(names, feature_map, kernel)):
                fm_grid = utils.make_grid(fm[:n], r)
                writer.add_image(f'[Conv{id}]{name}/feature_map', fm_grid)
                if id == 0: # 只能可视化3层的卷积核
                    kn_grid = utils.make_grid(kn[:n], r)
                    writer.add_image(f'[Conv{id}]{name}/kernel', kn_grid)

        else:
            print('There is nothing about the analysis of this model.')

        writer.close()


if __name__ == '__main__':
    """ toggle model """
    model_name = 'resnet'
    record_path = r'temp\train_progress\resnet\BS64LR0.01\record.pkl'
    # model_name = 'alexnet'
    # record_path = r'temp\train_progress\alexnet\BS64LR0.01\record.pkl'
    # model_name = 'vgg'
    # record_path = r'temp\train_progress\vgg\BS64LR0.01\record.pkl'

    writer.clear_runs()

    with open(record_path, 'rb') as f:
        record = pickle.load(f, encoding='bytes')
    state_dict = record['best_record']['weight']

    model = initialize_model(model_name, 102, feature_extract=False, use_pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)

    img_path = r'images\rose.jpg'
    # img_path = r'images\sunflower.jpg'
    # img_path = r'images\lotus.jpg'

    flower_classifier = Classifier(model)
    flower_classifier.predict(img_path)
    flower_classifier.analyze(img_path)
