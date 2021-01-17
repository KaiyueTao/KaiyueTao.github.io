---
title: Deep Learning/Machine Learning概念总结
layout: post
author: Kaiyue Tao
tags: [Leaning, Machine Learning, Deep Learning]
math: true
mermaid: true
---

**ML/DL中的一些常用概念和问题**

1. 梯度下降优化

    **SGD：**随机梯度下降，一般每次计算mini-batch，然后参数更新，完全取决于当前batch的梯度

    **Momentum：**顾名思义，模拟动量概念，积累之前前一次的“动量”，公式为：
    $$
    \begin{array}{l}m_{t}=\mu * m_{t-1}+g_{t} \\ \Delta \theta_{t}=-\eta * m_{t}\end{array}
    $$
    好处：在下降初期，加上了上一次的参数的比重，能够加速；在下降中后期，局部最小值震荡，梯度g趋近于0，此时系数 $\mu$可以增大更新幅度，跳出局部；梯度方向改变，$\mu$可以减少更新，抑制震荡。

    **Adagrad：**对学习率做了一个约束，公式为：
    $$
    \begin{array}{l}n_{t}=n_{t-1}+g_{t}^{2} \\ \Delta \theta_{t}=-\frac{\eta}{\sqrt{n_{t}+\epsilon}} * g_{t}\end{array}
    $$
    使得前期$g_t$较小的情况下，放大梯度；后期$g_t$较大的情况下，约束梯度，达到训练的稳定。善于处理稀疏数据。

    **Adadelta：**Adagrad的扩展，公式为：
    $$
    \begin{array}{l}n_{t}=\nu * n_{t-1}+(1-\nu) * g_{t}^{2} \\ \Delta \theta_{t}=-\frac{\eta}{\sqrt{n_{t}+\epsilon}} * g_{t}\end{array}
    $$
    处理后，不依赖全局学习率：
    $$
    \begin{array}{l}E\left|g^{2}\right|_{t}=\rho * E\left|g^{2}\right|_{t-1}+(1-\rho) * g_{t}^{2} \\ \Delta x_{t}=-\frac{\sqrt{\sum_{r=1}^{t-1} \Delta x_{r}}}{\sqrt{E\left|g^{2}\right|_{t}+\epsilon}}\end{array}
    $$
    一个特例为RMSprop，$\rho = 0.5$时，适合处理非平稳目标。所以RMSprop和Adadelta都是Adagrad的扩展。

    **Adam：**利用梯度一阶矩估计和二阶矩估计，动态调整学习率，他的思路和Adagrad很像，但是又加入了一阶矩。公式：
    $$
    \begin{array}{l}m_{t}=\mu * m_{t-1}+(1-\mu) * g_{t} \\ n_{t}=\nu * n_{t-1}+(1-\nu) * g_{t}^{2} \\ \hat{m}_{t}=\frac{m_{t}}{1-\mu^{t}} \\ \hat{n}_{t}=\frac{n_{t}}{1-\nu^{t}} \\ \Delta \theta_{t}=-\frac{\hat{m}_{t}}{\sqrt{\hat{n}_{t}}+\epsilon} * \eta\end{array}
    $$

    可以看这篇，讲的很全面：[Optimizing-gradient-descent](https://ruder.io/optimizing-gradient-descent/)

2. 不同激活函数
   
    **Sigmoid：** 
        $\sigma(x)=\frac{1}{1+e^{-x}}$
        $x:(-\infty, +\infty), y:(0,1)$
        特点：导数：$g'(z) = g(z)(1-g(z))$
        问题：饱和容易导致梯度消失
    
    **Tanh:**
        $\tanh (x)=2 \sigma(2 x)-1$
        $x:(-\infty, +\infty),y:(-1,1)$

    **ReLU:**
        $f(x)=\max (0, x)$
        问题：“死掉”，因为一旦小于0，就直接梯度消失了

    **Leaky ReLU:**
        $f(y)=\max (\varepsilon y, y)$
        解决了ReLU的脆弱性问题。

    **Maxout:**
        $f(x)=\max \left(w_{1}^{T} x+b_{1}, w_{2}^{T} x+b_{2}\right)$
        具有ReLU的优点，又不容易godie。但是每个神经元的参数都double了。

3. 最大似然估计/最大后验概率

4. MLE、MAP、Bayesian

    对于给定的x，求y，则要求的参数能够使得P(y|x)达到最大。
    **MLE：**极大似然估计，从训练集中找出使得分布的概率最大的参数。也就是由结果去推参数。使得似然函数最大。（i.i.d即所有采样是独立同分布的）
    **MAP：**最大后验估计，加上Prior即先验概率，因为贝叶斯派认为不存在具体的参数，只能给出一个参数的概率分布，也就是$P(D|\theta)P(\theta)$最大，这个可以结合贝叶斯公式去理。所以当数据集趋向于无穷大的时候，MAP也趋向于MLE（或者$P(\theta)$本就是均匀分布。
    **Bayesian：**贝叶斯估计，评估话语权。

    知乎这篇[MLE、MAP\Bayesian](\https://zhuanlan.zhihu.com/p/72370235) 和[详解MLE、MAP及贝叶斯公式理解](https://zhuanlan.zhihu.com/p/48071601)讲的很好。

5. 交叉熵

    **cross-entropy:** 其实就是似然值取对数后再归一化，然后取相反数。最后得到的就是$-p\ln p$的形式，固有cross-entropy loss。

6. Logistic Regression

    本质线性回归，可再通过sigmoid映射到(0,1)

    损失函数：
    $\mathrm{J}(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y_{i} \log \left(h_{\theta}\left(x_{i}\right)\right)+\left(1-y_{i}\right) \log \left(1-h_{\theta}\left(x_{i}\right)\right)\right]$

    透过这个公式其实很好理解，相当于对于groundtruth为0和1的情况分别讨论，用log做一个惩罚机制，如gt=1，y'=0时，则log0->∞。

    看这个[github](https://github.com/NLP-LOVE/ML-NLP)有各类总结，不过偏向NLP。

7. Random Forest

    **bagging**思想，从总体样本随机取出一部分样本训练，多次后，vote得到平均值输出。

    故随机森林就是每次随机且放回地从训练集抽取N个样本，训练一棵树，最后再进行投票制。

8.  Trainset不均衡的时候，Accuracy是不准确的。

To be continued...