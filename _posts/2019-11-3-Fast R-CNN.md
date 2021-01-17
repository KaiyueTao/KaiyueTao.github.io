---
title: Fast R-CNN
layout: post
categories: [learning]
author: Kaiyue Tao
tags: [Computer Vision, Learning, Object Detection]
math: true
mermaid: true
pin: true
---

**Author:** Ross Girshick from **Microsoft Research**

available at [Here](https://arxiv.org/abs/1504.08083)

## look back at R-CNN(also SPPnet)

本文是在R-CNN的基础上的进一步研究（同一作者），故回顾了R-CNN的一些缺陷：

1. Training过程是多阶段的：首先训练ConvNet，然后再用CNN提取的feature训练SVM，用SVM代替原本的softmax，最后再进行bbox regressors的训练。
2. Training is expensive，features 从每个proposal提取，再写入disk，耗费时间和空间。
3. 目标检测太慢，原因：a ConvNet forward pass for each object proposal，没有共享计算。（SPPnet则使用了shared computation：首先计算整个图的feature，再从中提取每个proposal的feature，则无需每次重复计算feature。然后采用max-pooling的操作将不同size的proposal提取6*6的最终feature，比R-CNN在test阶段快10-100倍，在train阶段快3倍）
   
当然，SPPnet的train也是multi-stage的，且不同于R-CNN，fine-tuning无法更新卷积层，使得accuracy收到了限制。

## 主要贡献

1. 更高的mAP（与R-CNN、SPPnet相比）
2. 使用multi-task loss，train一步到位
3. train阶段可以更新所有层的参数
4. 无需额外的存储空间（for feature caching）

## 网络结构

<img src="https://cdn.mathpix.com/snip/images/6vYlYyQbfuqq5HOubOwe2kLie-BO0n8RSVvlMvg6cIw.original.fullsize.png" />

首先让整个image通过CNN计算得到feature map，然后对于生成的每个region proposal，用一个RoI Pooling层提取等长的feature vector。

再将feature vector输入一个fc层，通过两个分支，得到两个output：

- k+1个类（k+background）的概率p，由softmax计算
- k个类的bbox regression offsets，用(t<sub>x</sub>,t<sub>y</sub>,t<sub>w</sub>,t<sub>h</sub>)表示

**RoI pooling layer**
   
将RoI（Region of Interest）表示为：

<center> (r,c,h,w) </center>

即左上角的坐标和宽高值。将RoI划分为H &times; W个grid，则每个grid的大小为h/H &times; w/W，然后运用max pooling将每个grid的max值提取，使得最终得到相同的size（H&times;W），保证了尺度不变性。

## Training

1. Initialize from pre-trained networks
    
    替换一些层（RoI、最后的sibling layers）

2. <span id="fune-tuning">Fune-tuning for detection</span>

    为了让train更为高效，在组成batch时，首先选择N张图，然后选择一张图的R/N个RoI。这样同一张图的RoI可以共享计算。在实际选取的时候，使N尽量小，则可以更大程度地实现共享。

    文章使用了N=2，R=128的超参数组合，比128个不同图片的RoI训练起来快64倍。

3. Multi-task loss

    u为ground-truth class，v为ground-truth bbox regression：
   
    L(p,u,t<sup>u</sup>,v) = L<sub>cls</sub>(p,u)+&lambda;[u&ge;1]L<sub>loc</sub>(t<sup>u</sup>,v)

    其中：
    
    - L<sub>cls</sub>(p,u)=-logp<sub>u</sub>，表示true class的possibility的loss。
    
    - [u&ge;1]当u&ge;1时输出为1，其他则为0。其中background class=0，则不计算L<sub>loc</sub>，其他时候则考虑L<sub>loc</sub>。

    - L<sub>loc</sub>: L<sub>loc</sub>(t<sup>u</sup>,v) = $\sum_{i=x,y,w,h}$ smooth<sub>L<sub>1</sub></sub>(t<sub>i</sub><sup>u</sup> - v<sub>i</sub>)

    $$
    smooth_{L_1}(x) =
    \begin{cases}
    0.5x^2 &if |x|<1 \\
    |x|-0.5 &otherwise  
    \end{cases}
    $$

    - 而&lambda;则控制了两个loss的平衡，在文章的实验中都使用了&lambda;=1这个参数。

    **就我个人理解而言，这样设置Multi-Loss的原因大概是综合考虑classifier和bbox regression的loss，使得back-propagation对所有参数都有效，则实现了single-stage train。**

4. Mini-Batch sampling

    [上面](#fune-tuning)提到，为了使得train更高效，每个SGD mini-batch都由128个RoI from 2 Images组成，其中25%的IoU>0.5，它们的class&&ge;1，而剩下的RoIs则IoU在0.1～0.5之间，标记为u=0（即background）。

## Detection

检测的具体流程如下：

- 输入一张图片以及R个Proposals（文章测试了不同数目的R）。
- 对每个RoI，forward pass输出class probability，以及一系列（k个class分别）的bounding-box offset，然后用R-CNN中的[non-maximum](/2019-11-1-R-CNN.md)算法得到最终的bbox。

## Result

1. Fast training and testing, test runtime: about 0.3s
2. state-of-the-art mAP:
    - on PASCAL VOC 2012: 65.7%
    - on VOV 2007: 66.9% (68.1% after remove "difficult" example)
3. Fune-tuning conv layers in VGG16, improves mAP:
    not all layers should be fine-tuned，文章发现在VGG16中，只有conv3_1之后的层需要fune-tune（9 of 13），则减少了训练时间以及GPU memory占用率。