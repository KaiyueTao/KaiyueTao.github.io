---
layout: post
title: Retina Net (Focal Loss)
author: Kaiyue Tao
tags: [Computer Vision, Learning, Object Detection]
math: true
mermaid: true
pin: true
---

**Focal Loss for Dense Object Detection**

**Author:** Tsung-Yi Lin etc from **Facebook AI Research**

available at [Here](https://arxiv.org/abs/1708.02002)

## 主要贡献

1. 提出了Focal Loss
2. 使用Focal Loss的RetinaNet，在速度上比肩one-stage detector，在accuracy上超过所有的detector

## Class Imbalance

这是文章的出发点：

很多one-stage的探测方法，在训练时，对于一张图计算10<sup>4</sup>-10<sup>5</sup>的候选区，但真正含有目标物的却非常少，这会导致两个问题：

- training is inefficient，因为easy negatives对learning没有有用的贡献
- 太多的easy negatives充满了整个训练阶段，会主导learning的方向，产生非常差的模型

通常的解决方法是用一些hard examples，在设计训练集时分配，而Focal Loss则可以直接解决这些问题。

概括地说，focal loss被设计成在loss层面上使得easy examples对learning的贡献非常小。

## Focal Loss

1. 公式：
   
    $$
    FL(p_{t}) = - (1 - p_{t})^{\gamma}log(p_{t})
    $$

2. 特点：

    - 当样本是misclassfied且p<sub>t</sub>很小的时候，modulating factor$(1 - p_{t})^{\gamma}$接近1，loss不受影响。当p<sub>t</sub>接近1的时候，modulating factor趋近于0，则这一well-classfied的loss也会很小。
    - $\gamma=0$的时候，FL与CE相同，文章发现在实验中取$\gamma=2$时效果比较好。

    总的来说，FL通过这个modulating factor降低了来自easy examples的贡献，并且可以将low loss的范围扩展。如$\gamma=2$时，$p_{t}=0.9$时loss也比CE要低100倍，这样的话就会迫使网络纠正missclassfied的样本，因为对于$p_{t}<0.5$的样本，loss至多也就减小了4倍而已。

3. $\alpha$-balanced variant

    在实际中又加上了：

    $$
    FL(p_{t}) = - \alpha_{t} (1 - p_{t})^{\gamma}log(p_{t})
    $$

    因为在实验中发现这样比原来的loss可以再提高一点轻微的accuracy（LOL）。并且文章还注意到，在loss层结合对p做sigmoid效果更好。

## RetinaNet网络结构

<img src="https://cdn.mathpix.com/snip/images/jLOjcMKLml6F8J-AEx5RF2lAWKhQUr9PzKSzXju1-d0.original.fullsize.png" />

由Res-Net+FPN得到多尺度的卷积提取特征，然后输入两个subnet(FCN)，一个用于分类，一个用于bbox regression。

两个subnet之间结构相同，但是不共享参数，文章发现这样的设计比超参数的选取更为重要。

## Training

1. loss
   
    使用了focal loss作为classification subnet的损失函数，对于每一张图片，计算其所有anchors的loss，并求和得到一个sum loss，再除以与ground-truth box相指定的anchors数目。

    这里不直接除以所有anchors数目，而是选择了与groun-truth box指定的anchors数目，是因为大多数anchors都是easy negative例，focal loss非常小，所以可以忽略不计。这样也使得loss更为关注hard examples部分。

2. hyper-parameters

    $\alpha$的选择要考虑到$\gamma$，所以这里需要进行微调（$\alpha$随着$\gamma$的增加而轻微降低），最终实验得到的最佳组合是：

    $\alpha = 0.25, \gamma = 2$

    文章发现RetinaNet对于$\gamma \in [0.5,5]$的效果都相对较好。

3. Initialize

    所有的conv layer除了最后一层，都初始化为$b=0$、Gaussian weight $\sigma=0.01$，而最后一层的bias初始化为：

    $b=-log((1-\pi)/\pi)$

    其中$\pi$表示anchor一开始的confidence为标为$\pi$，这是为了一开始训练时的稳定性，因为一开始的class imbalence会导致有大量background anchors会形成不稳定的loss值。

4. Optimization

    - SGD
    - trained for 90k iterations，initial learning rate = 0.01，之后在60k除以10，在80k再除以10。
    - weight decay = 0.0001, momentum = 0.9
    - focal loss用于classfy，而smooth $L_1$ loss则用于regression

training time大概在10-35小时之间。

## Results

在COCO上AP达到39.1，值得注意的是small size和medium size的效果有很大提升。







