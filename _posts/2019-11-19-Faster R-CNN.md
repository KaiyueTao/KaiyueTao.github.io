---
layout: post
title: Faster R-CNN
author: Kaiyue Tao
tags: [Computer Vision, Learning, Object Detection]
math: true
mermaid: true
pin: true
---

**Arthur:** Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun

available at [Here](https://arxiv.org/abs/1506.01497)

## 主要贡献

An elegant and effective solution: RPN，使得proposals的计算时间达到了每张图10ms。

## 出发点

- proposal are the test-time computational bottleneck in detection systems

    在CPU下一张图的Selective Search阶段要花费2s。且Region proposal method只能在CPU上跑，这使得之后CNN的计算虽然能够利用GPU，但是相比之下节省的时间是相当有限的。

## 网络结构

<img src="https://cdn.mathpix.com/snip/images/8wBuWzahVLUN8zR6H6_U7WwFUr4NWVaNd50U5074Soc.original.fullsize.png" />

事实上主要的改变还是RPN，其他的部分如conv layers计算feature map、RoI pooling层，以及之后的classfier部分都和Fast R-CNN类似。

如图所示，RPN与object detection network是共享cnn计算的，故feature maps只需计算一次，并同时作为RPN和RoI的输入。

## Region Proposal Network

<img src="https://cdn.mathpix.com/snip/images/1MaRllS4xeKXQ1Wf0L7CCmK46XiFMvbeEoJdXLpD_p8.original.fullsize.png" />

将$n \times n$的sliding windows应用在输入的feature map上，每一个sliding windows计算的结果都转换成256-d（网络为ZF，若用VGG则为512d）的向量，并作为之后的FCN层的输入（box-regression layer(reg layer)和box-classification layer(cls layer)）。

1. Anchors

    对于每个sliding window，预测k个proposals，故cls layer输出2k个分数（two-class softmax layer），而reg layer输出4k个坐标（k个bbox）。

    $H \times W$尺寸的图一共生成WHk个anchors。在文章中使用了k=9的超参。

2. Translation Invariant

    所谓**平移不变性**，即是指当图像中的物体发生平移时，他对应的proposal跟着变化，预测的function应该能够具备预测这种变化后的proposal的能力。（MultiBox方法采用kmeans聚类，就无法保证这种不变性。

    Faster R-CNN则保证了平移不变性。其参数为$512 \times (4 + 2) \times 9$(for VGG-16)，这比MultiBox的参数少了很多，因此也减少了在小数据集上过拟合的可能。

3. Multi-scale anchors 
   
    两种方式：

    - image/feature pyramids：即输入不同尺寸的image，对每个尺寸计算feature map，这个方法的弊端是时间代价大
    - pyramid of filters：运用不同尺寸的sliding window

    两种方式通常被结合起来运用。

    而文章提出了一种新的方法：pyramid of anchors，只需要改变anchors的不同尺度，对于image和sliding window都只需要一个尺寸即可。

4. Loss Function

    定义了两种positive label：

    - 与某一groundtruth box的IoU最大的anchor
    - 或者与所有的groundtruth box的IoU都$>0.7$

    negative label：

    - 与所有groundtruth box的IoU都$<0.3$

    故loss function的定义为：

    $$
    L({p_i},{t_i}) = \frac{1}{N_{cls}} \sum L_{cls}(p_i,p_i^*) + \lambda \frac{1}{N_{reg}} \sum p_i^* L_{reg} (t_i,t_i^*)
    $$

    （i代表anchor在mini-batch中的index；p在anchor为正时=1，否则=0；而t则是4-d的向量，代表bbox。）

    其中：$L_{reg}(t_i,t_i^*) = R(t_i - t_i^*)$，$R$即是$smooth L_1$，这个loss只有在anchor为positve时被激活，当anchor为negative，则为0。

    t的向量四个值表示为：

    $$
    \begin{aligned} t_{\mathrm{x}} &=\left(x-x_{\mathrm{a}}\right) / w_{\mathrm{a}}, \quad t_{\mathrm{y}}=\left(y-y_{\mathrm{a}}\right) / h_{\mathrm{a}} \\ t_{\mathrm{w}} &=\log \left(w / w_{\mathrm{a}}\right), \quad t_{\mathrm{h}}=\log \left(h / h_{\mathrm{a}}\right) \\ t_{\mathrm{x}}^{*} &=\left(x^{*}-x_{\mathrm{a}}\right) / w_{\mathrm{a}}, \quad t_{\mathrm{y}}^{*}=\left(y^{*}-y_{\mathrm{a}}\right) / h_{\mathrm{a}} \\ t_{\mathrm{w}}^{*} &=\log \left(w^{*} / w_{\mathrm{a}}\right), \quad t_{\mathrm{h}}^{*}=\log \left(h^{*} / h_{\mathrm{a}}\right) \end{aligned}
    $$

    其中x、y、w、h分别是box的中心和宽高。

5. Training

    为了防止出现bias，在训练时随机选取256个anchors，且正负比为1:1，组成mini-batch。

    - learning_rate = 0.001 for 60k mini-batches and 0.0001 for next 20k

    - momentum = 0.9, weight_decay = 0.0005