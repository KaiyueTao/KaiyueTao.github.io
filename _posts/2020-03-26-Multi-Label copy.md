---
layout: post
title: Multi-label Classification[Paper总结]
author: Kaiyue Tao
tags: [Computer Vision, Learning, Multi-label Classification]
math: true
mermaid: true
pin: true
---

![](https://tva1.sinaimg.cn/large/008eGmZEly1gmquc8hd4wj30n8096dhf.jpg)

对MLC问题的解决主要从两个角度入手：
- Attention：16年的CNN-RNN通过RNN完成隐含的注意力转移，近两年随着attention的发展，出现的paper则通过一些learning机制来迫使网络学习更好的attention。
- GNN/GCN：MLC可以说是最适合图网络应用的任务之一了，一般网络对同一张图片提取不同类别的特征，然后根据邻接矩阵进行信息聚合交互，通常Adjacency Matrix由trainset中不同标签的关联度（即同时出现的概率）来定义

## Attention

### CNN-RNN [CVPR 2016]

这篇采用RNN时序模型来对图片做一个label序列的预测，利用了RNN的hidden语义，结合之前的label信息得到output，与CNN得到的feature map结合，最后预测下一个label。

具体笔记： [Here](/posts/CNN-RNN/)

### Learning Spatial Regularization with Image-level Supervisions for Multi-label Image Classification

学习图像空间的注意力，得到attention map和confidence map，来增强feature，提升一些类别的confidence。

[PDF](https://arxiv.org/abs/1702.05891)

### Visual Attention Consistency under Image Transforms for Multi-Label Image Classification

很有意思的想法，通过对图像进行变换（image transform）并迫使其feature一致，来增强attention。

[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Visual_Attention_Consistency_Under_Image_Transforms_for_Multi-Label_Image_Classification_CVPR_2019_paper.pdf)

## Graph Neural Network

### Learning Semantic-Specific Graph Representation for Multi-Label Image Recognition

引入标签semantic的语义来提取不同类别的feature，然后通过GNN进行信息交互。semantic特征是从GloVe学习得到的feature，但我在实验的时候发现semantic feature其实没那么重要，随机化一个feature也可以到差不多的performance，所以这个semantic这个点其实是有疑问的。个人认为只要对不同的类别提取不同的特征即可，引入语义的效果有待商榷。

[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.pdf)

### Multi-Label Image Recognition with Graph Convolutional Networks

旷视的这篇也是非常经典的用了GCN来增强关联信息，但不同的是这里visual feature只提取一次，而GCN则是标签语义semantic feature间的图卷积，以训练一个分类器。最后与visual feature点乘得到class score。这里同样个人感觉semantic feature的选取没那么重要，关键还是在于adjacency mat的生成。

[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf)

### Mind Your Neighbours: Image Annotation with Metadata Neighbourhood Graph Co-Attention Networks

一篇挺有意思的文章，引入image caption的信息，用knn找到邻近图片来辅助标注。其思想是两张邻近的图片中含有相似的目标，通过图网络交互从而来引导attention。

[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Mind_Your_Neighbours_Image_Annotation_With_Metadata_Neighbourhood_Graph_Co-Attention_CVPR_2019_paper.pdf)

### ppt

Find my ppt for MLC problem on the lab meeting.

[MLC.pptx](/assets/MLC.pptx) [MLC2.pptx](/assets/MLC2.pptx)