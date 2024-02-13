![cover](/images/deploy.png "cover")

# LearningNN['main']

main branch基于Pytorch（2.2.0）实现了三种经典的卷积神经网络（CNN）：AlexNet、Vgg和ResNet。数据集选用torchvision自带的flower102数据集，在预训练模型的基础上调整最后的全连接层实现花卉分类。此项目包含完整代码，在环境配置正确的情况下能够开箱即用。

## 环境配置

* Windows 11
* Python 3.8.18
* Pytorch 2.2.0
* CUDA 11.8

## 文件说明

* view_data.py：加载数据集，预处理，数据预览
* model.py：加载模型，冻结除最后全连接层之外的层
* train.py：训练模型，输出并保存训练过程及结果
* test.py：测试模型
* deploy.py：模拟部署模型，输入单张图片，预测类别
* cat_to_name.json：类别id与name之间的映射
