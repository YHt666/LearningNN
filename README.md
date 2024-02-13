![cover](/images/cover.png "cover")

# LearningNN['tensorboard']

[tensorboard branch](https://github.com/YHt666/LearningNN/tree/tensorboard)在[main branch](https://github.com/YHt666/LearningNN)的基础上加入了tensorboard可视化，便于动态观察训练过程以及加深对网络的理解。

## tensorboard可视化

定义了Writer类用于数据传输。

* 训练数据预览——见view_data.py
* model graph （模型结构图）——见train.py
* 动态显示训练过程的loss、accuracy、weights——见train.py
* 各类别的PR curve——见test.py
* 训练完成后卷积层的feature map和kernel——见deploy.py

## 环境配置

* Windows 11
* Python 3.8.18
* Pytorch 2.2.0
* CUDA 11.8

## tensorboard使用注意事项

* 在命令行输入

    ```cmd
    tensorboard --logdir=runs --port=6006  --reload_multifile=true
    ```

    启动tensorboard，实测最后的“--reload_multifile=true”不能少，否则tensorboard不能自动加载新生成的events （尝试过加“--reload_intervel=30”， 不work）
* 如果端口被占用，可以更改端口号，或者解除占用：

  * 命令行输入

    ```cmd
    netstat -ano|findstr "6006"
    ```

    可以得到占用端口的进程的pid

    ```cmd
    TCP    127.0.0.1:6006         0.0.0.0:0              LISTENING       2960
    ```

  * 杀死该进程

    ```cmd
    taskkill /f /pid 2960
    ```
