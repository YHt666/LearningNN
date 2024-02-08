import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'        # 指定GPU
os.environ['TORCH_HOME']='E:/PretrainedModel'   # 修改预训练模型下载路径

import torch
thread = 8 #设到10以下
torch.set_num_threads(int(thread))

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import matplotlib.pylab as plt

from view_data import data
from model import initialize_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def valid_model(model, valid_dataloader, loss_fn):
    model.eval()
    size = len(valid_dataloader.dataset)
    num_batches = len(valid_dataloader)
    valid_loss = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in valid_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = model(imgs)
            valid_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    valid_loss /= num_batches
    correct /= size

    model.train()
    return valid_loss, correct


def train_model(model, dataloader, loss_fn, optimizer, scheduler, params):
    train_loss_list = []
    valid_loss_list = []
    valid_acc_list = []
    total_step = 0
    best_acc = 0
    best_record = {}

    model.train()
    for i in range(params['epoch']):
        progress_bar = tqdm(dataloader['train'], desc=f'Epoch {i}')
        total_loss = 0
        batch_count = 0
        for imgs, labels in progress_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)  # 采用indices和one-hot均可，此处选择前者
            pred = model(imgs)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / batch_count,
                'lr': optimizer.param_groups[0]['lr'],
                                      })
            
            total_step += 1
            train_loss_list.append((total_step, loss.item()))

            # evaluation
            if total_step % params['eval_interval'] == 0:
                valid_loss, correct = valid_model(model, dataloader['valid'], loss_fn)
                tqdm.write(f'\nEval: Loss: {valid_loss}, Accuracy: {100*correct}%')
                valid_loss_list.append((total_step, valid_loss))
                valid_acc_list.append((total_step, correct))
                if correct > best_acc:
                    best_record = {
                        'step': total_step,
                        'weight': model.state_dict(),
                        'acc': correct,
                    }
                    best_acc = correct

            # save model
            if total_step % params['save_interval'] == 0:
                torch.save(
                    {
                        'model': model.state_dict(),
                    },
                    os.path.join(params['save_path'], f'checkpoint_{total_step}.pth')
                )

        scheduler.step()
    return train_loss_list, valid_loss_list, valid_acc_list, best_record


if __name__ == '__main__':
    # 训练参数设置
    params = {
        'model_name': 'vgg',  # 选择官方提供的模型，可选['resnet', 'alexnet', 'vgg']
        'lr': 1e-2,
        'batch_size': 64,
        'epoch': 10,
        'eval_interval': 50,   # 每隔多少batch评估一次
        'save_interval': 100,   # 每隔多少batch保存一次
    }
    # 模型文件保存根目录
    save_path = f'temp/model/{params["model_name"]}/BS{params["batch_size"]}LR{params["lr"]}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    params['save_path'] = save_path

    print(f'pytorch version {torch.__version__} with cuda {torch.version.cuda}')
    print(f'device: {device}')

    model_name = params['model_name']   
    # 是否使用已训练好的特征提取层
    feature_extract = True
    model = initialize_model(model_name, 102, feature_extract, use_pretrained=True)
    model = model.to(device)
    print(model)

    # 查看训练的层
    print('params to learn:')
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print('\t', name)

    optimizer = torch.optim.Adam(params_to_update, lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()

    batch_size = params['batch_size']
    dataloader = {x: DataLoader(data[x], batch_size=batch_size, shuffle=True, 
                                num_workers=2, pin_memory=True) for x in ['train', 'valid']}
    
    while True:
        if input('Start train? (y/n)') == 'y':
            break

    train_loss_list, valid_loss_list, valid_acc_list, best_record = train_model(
        model, dataloader, loss_fn, optimizer, scheduler, params
    )
    print(f'Best record:\nStep: {best_record["step"]}, Acc: {best_record["acc"]}')

    # save training progress
    save_path = f'temp/train_progress/{params["model_name"]}/BS{params["batch_size"]}LR{params["lr"]}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    record = {
        'train_loss_list': train_loss_list,
        'valid_loss_list': valid_loss_list,
        'valid_acc_list': valid_acc_list,
        'best_record': best_record,
    }
    with open(os.path.join(save_path, 'record.pkl'), 'wb') as f:
        pickle.dump(record, f)

    # visualize loss curve
    fig, ax = plt.subplots(2, 1, figsize=(6, 8), dpi=100, sharex='all')
    ax[0].plot(*zip(*train_loss_list), label='train')
    ax[0].plot(*zip(*valid_loss_list), label='valid')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[1].plot(*zip(*valid_acc_list))
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Iterations')
    plt.savefig(os.path.join(save_path, 'loss_acc_curve.jpg'))
    plt.show()
    