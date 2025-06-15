# AI框架与科学计算83组大作业

## 任务

本次大作业科学任务为基于条件生成对抗网络（GAN）实现流体流超分辨率及时间连贯性生成，复现论文tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid Flow

## 方法简述

论文提出tempoGAN 框架，与传统生成对抗网络不同的是，基于流体超分辨率的任务特性，设计由生成器（Generator, G）、空间判别器（Spatial Discriminator, Ds） 和 时间判别器（Temporal Discriminator, Dt） 组成的tempoGAN网络框架，保证生成流行的时间一致性。

## 实验数据

本实验使用数据集为使用开源代码包 mantaflow 生成的 2d 流体数据集。训练集中共16组流体模拟视频，每组包括200帧的流体演变序列。序列包括低分辨率流体图像（128，128），和高分辨率流体图像（512，512），所有图像均为灰度图。

## 代码结构



## 代码详解

tempoGAN网络核心为生成器（Generator）、判别器（Discriminator）、时间判别器（Discriminator_tempo），下面分别展示代码。

### 生成器

生成器输入为低分辨率流体数据，输出为高分辨率流体数据。

由于神经网络输入首先将数据插值为分辨率为（256，256）的流体数据，因此需要对输入数据进行转化，代码如下：

```python
def transform_in(self, _in):
        ratio = 2
        input_dict = reshape_input(_in)
        density_low = input_dict["density_low"]
        density_low_inp = interpolate(density_low, ratio, "nearest")
        return {"input_gen": density_low_inp}
```

生成器为四层的残差网络（ResNet）:

```python
# define Generator model
    model_gen = ppsci.arch.Generator(**cfg.MODEL.gen_net)
    model_gen.register_input_transform(gen_funcs.transform_in)
    disc_funcs.model_gen = model_gen
```

生成器采用l1和l2损失和逐层损失:

```python
# l1 loss
        loss_l1 = F.l1_loss(
            output_dict["output_gen"], output_dict["density_high"], "mean"
        )
        losses = loss_l1 * self.weight_gen[0]

        # l2 loss
        loss_l2 = F.mse_loss(
            output_dict["output_gen"], output_dict["density_high"], "mean"
        )
        losses += loss_l2 * self.weight_gen[1]
```

### 判别器

判别器输入为低分辨率流体数据与真实高分辨率流体数据或生成高分辨率流体数据的拼接。

```
disc_net:
    input_keys:  ['input_disc_from_target', 'input_disc_from_gen']  # 'NCHW'
    output_keys:  ['out0_layer0', 'out0_layer1', 'out0_layer2', 'out0_layer3', 'out_disc_from_target', 'out1_layer0', 'out1_layer1', 'out1_layer2', 'out1_layer3', 'out_disc_from_gen']
    in_channel:  2
    out_channels:  [32, 64, 128, 256]
    fc_channel:  1048576
    kernel_sizes:  [[4, 4], [4, 4], [4, 4], [4, 4]]
    strides:  [2, 2, 2, 1]
    use_bns:  [false, true, true, true]
    acts:  ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', null]
```

### 时间判别器

时间判别器输入为3帧连续的真实高分辨率流体数据或生成高分辨率流体数据。

```
tempo_net:
    input_keys:  ['input_tempo_disc_from_target', 'input_tempo_disc_from_gen']  # 'NCHW'
    output_keys:  ['out0_tempo_layer0', 'out0_tempo_layer1', 'out0_tempo_layer2', 'out0_tempo_layer3', 'out_disc_tempo_from_target', 'out1_tempo_layer0', 'out1_tempo_layer1', 'out1_tempo_layer2', 'out1_tempo_layer3', 'out_disc_tempo_from_gen']
    in_channel:  3
    out_channels:  [32, 64, 128, 256]
    fc_channel:  1048576
    kernel_sizes:  [[4, 4], [4, 4], [4, 4], [4, 4]]
    strides:  [2, 2, 2, 1]
    use_bns:  [false, true, true, true]
    acts:  ['leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', null]
```

## 运行说明

本次代码基于PaddlePaddle以及PaddleScience，请安装后运行，安装代码如下：
1. 创建环境

```
conda create -n ppsci_py39 python=3.9
```

2.启动环境

```
conda activate ppsci_py39
```

3.访问PaddlePaddle官网安装环境

https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html

4.安装PaddleScience

```
python -m pip install -U paddlesci -i https://pypi.tuna.tsinghua.edu.cn/simple
```

5. 下载数据集

```
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_train.mat -P datasets/tempoGAN/
wget -nc https://paddle-org.bj.bcebos.com/paddlescience/datasets/tempoGAN/2d_valid.mat -P datasets/tempoGAN/
```

6.一键启动

## 实验内容

本次大作业对比了普通的生成对抗网络和包含时间判别器的tempoGAN方法，经实验验证，tempoGAN生成图像质量更优，指标见下表.

|          | MSE                 | PSNR            | SSIM             |
| -------- | ------------------- | --------------- | ---------------- |
| GAN      | $10.87e^{-5}$     | 39.64           | 0.9946           |
| tempoGAN |  *$4.21e^{-5}$* | **47.19** | **0.9974** |
