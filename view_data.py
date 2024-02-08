""" 数据可视化 """
#%%
# import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from torchvision.transforms import v2 as transforms
import numpy as np
import matplotlib.pyplot as plt


#%%
# 数据集存储位置
data_dir = 'E:/Dataset'

# 第三方提供的数据库图像的均值与标准差
meta_data = {}
meta_data['im_mean'] = [0.485, 0.456, 0.406]
meta_data['im_std'] = [0.229, 0.224, 0.225]

# 数据增强与标准化
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(45),  # 随机旋转，范围-45至45度
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5), # 以0.5的概率随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),   # 随机垂直翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 调整亮度、对比度、饱和度、色相
        transforms.RandomGrayscale(p=0.025),    # 概率转换为灰度图, 即R=G=B
        transforms.ToTensor(),  # 转换为tensor并归一化
        transforms.Normalize(mean=meta_data['im_mean'], std=meta_data['im_std']) # 标准化（第三方提供）
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=meta_data['im_mean'], std=meta_data['im_std'])
    ]),
}

data = {}
# 第三方的数据集划分test数据较多，此处将其作为train数据集
data['train'] = datasets.Flowers102(
    root=data_dir,
    split='test',
    download=True,
    transform=data_transforms['train'],
)
data['valid'] = datasets.Flowers102(
    root=data_dir,
    split='val',
    download=True,
    transform=data_transforms['valid']
)
data['test'] = datasets.Flowers102(
    root=data_dir,
    split='train',
    download=True,
    transform=data_transforms['valid']
)

def im_convert(tensor):
    """ 还原数据 """
    image = tensor.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array(meta_data['im_std']) + np.array(meta_data['im_mean'])
    image = image.clip(0, 1)    # 由于浮点操作过程中的误差，逆标准化后的像素值可能会超出[0,1]，故需钳位
    return image


#%%
if __name__ == '__main__':
    batch_size = 16

    dataloader = {x: DataLoader(data[x], batch_size=batch_size, shuffle=True, 
                                num_workers=2, pin_memory=True) for x in ['train', 'valid']}
    # print(f'train_data_size: {len(dataloader["train"].dataset)}')
    # print(f'valid_data_size: {len(dataloader["valid"].dataset)}')
    print(f'train_data_size: {data["train"].__len__()}')
    print(f'valid_data_size: {data["valid"].__len__()}')
    print(f'test_data_size: {data["test"].__len__()}')


    fig, ax = plt.subplots(4, 4, figsize=(10, 6), dpi=100)
    axf = ax.flat

    dataiter = iter(dataloader['train'])
    # dataiter = iter(dataloader['valid'])
    img, label = next(dataiter)

    for i in range(batch_size):
        axf[i].imshow(im_convert(img[i]))
        axf[i].set_title(f'{label[i]}')
    fig.tight_layout()
    plt.show()


# %%
