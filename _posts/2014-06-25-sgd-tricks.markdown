---
layout: post
title: "SGD Tricks"
date: 2014-06-25 13:46:13 -0400
comments: true
categories: Paper Optimization Notes
---

## 简介 ##

Bottou在Stochastic Gradient Descent和对Neural Network中的优化很有建树。
他早先是Lecun的同事，现在在Microsoft做研究。最近读了他的报告Stochastic
Gradient Descent Tricks。这个报告其实是一本Neutral Networks Tricks of
the Trade一本书的一章。这本书有电子版。Amazon上的一则评论说因为有不同的
人写的，所以有些对立观点。

SGD在大数据和NN优化Toolbox上乃是“装机必备”，比如PyLearn2, Caffe, Torch，
CXXNet。传说NN优化太多trick，只有在Lecun周围方圆百里的人才懂得如何调参。

<!--more-->

## 引用 ##

> Léon Bottou: Stochastic Gradient Tricks, Neural Networks, Tricks of
> the Trade, Reloaded, 430–445, Edited by Grégoire Montavon, Genevieve
> B. Orr and Klaus-Robert Müller, Lecture Notes in Computer Science
> (LNCS 7700), Springer, 2012.

## 公式 ##

SGD 迭代公式非常简单。一般的Gradeint Descent的迭代式是，详见Convex Optimization等书。
$$w_{t+1}  = w_t - \gamma \nabla f(w_t)$$
因为Machine Learning的问题基本都是在训练集上求loss function的最小值。于是这里的梯度就是
N个训练样本上的梯度。

$$w_{t+1}  = w_t - \gamma \sum_{i=1}^N \nabla f(x_i, w_t)$$

GD迭代一次使用全部训练集的梯度，SGD只使用一个训练样本的梯度而已。
$$w_{t+1}  = w_t - \gamma \nabla f(x_i, w_t)$$

文中列举了Linear regression，Perceptron，K-Means，SVM，Lasso等SGD迭代公式。

文中的一个误差公式有点意思。说的是误差是由1. Approximation Error近似
误差，Estimation Error估计误差和Optimization Error优化误差组成。这和
统计学习理论LFD一书中不大一样。这里取的期望应该是所有数据空间的。统计
学习理论的用样本误差加上一个以VC维和样本数量的函数作为推广误差界的。
推广误差界应该对应的是第一项近似误差。

为什么SGD而不是以前经典的GD被大量运用于现在的优化问题，我的理解是一方
面因为现在数据量太大放不进内存。另外一个理解以前的算法都以达到
预定accuracy的迭代次数为标准。比如GD是线性收敛，即以等比级数收敛，大概需要$\log(1/\rho)$的迭代次数。
但是把每次迭代的遍历数据的个数也考虑进去的话情况就不一样了。文中说明了SGD要比GD都要快的原因。

这里有个地方我不是很明白。$$\mathcal{E} = \mathcal{E}_{\text{app}} + \mathcal{E}_{\text{est}} + \mathcal{E}_{\text{opt}}$$
收敛的速度是这个三项里面最慢的一项决定的。可是第一项一旦hypothesis确定了，怎么还能在优化的时候减少呢？可能要看看是
另外NIPS的文章。

在一些convex假设下$\mathcal{E}_{\text{est}}$大概以$(\log (n) /
n)^{\alpha} $收敛。如果$n$趋向于无穷大我们有无限多的数据那么这个误差确
实趋向于0。又因为我上面没看懂的原因，这三个误差收敛速度一样。因为
$\rho$差不多等于上面这个收敛速度。于是得到了$n$与$\epsilon$的关系。但是
这个是$n$是关于$\epsilon$的超越函数所以要用脚注2的公式近似一下。

$$
\epsilon^{\frac{1}{\alpha}} \sim \frac{\log n}{n}
$$

得到$\log n \sim
\frac{1}{\alpha}\log \frac{1}{\epsilon}$再代入到原来的公式中得到

$$ 
n \sim \epsilon^{-1/\alpha}\log(1/\epsilon) 
$$

意思是说达到$\epsilon$这样
的误差需要多少样本，代入到表格2中的第3行就得到了GD需要
$\frac{1}{\mathcal{E}^{1/\alpha}} \log^2 \frac{1}{\mathcal{E}}$这样的时
间。而SGD因为每次迭代都只要一个数据，所以还是$\frac{1}{\mathcal{E}}$。

## Tricks ##

+ Randomly shuffle
+ User preconditioning techniques
需要看书1.4.3和1.5.3
+ Monitor both the training cost and the validation error
+ check the gradients using finite differences
+ Experiment with the learning rates using a small sample of the training set

+ Linear Models with $L_2$ regularization

$$w_{t+1} = (1-\gamma_t\lambda) w_t - \gamma_ty_tx_t\ell^\prime(y_tw_tx_t) $$

如果$x_t$很sparse的话，那么很多$w_t$的元素都只是在乘以一个数缩放而已，偶尔要相加第二项的数。如果大多数元素都需要做
每次迭代都要做差不多的缩放的话，不如把这些缩放都连乘，存在一个$s_{t+1}$这个变量里。这样就假设$w_t = s_tW_t$，带入上面的公式，
得到了$g_t, s_{t+1}, W_{t+1}$三个公式，尽管多了一个除法。

下面一个trick

$$ \gamma_t = \gamma_0(1+\gamma_0\lambda t)^{-1}$$

这里的$\lambda$用的是regularization项前面的$\lambda$系数。

+ ASGD把前面的迭代得到$w_t$都累加起来。这种累加可以写成迭代的形式，再考虑到$x$是sparse的情况。考虑到$w_t = s_tW_t$，而
$\bar{w}_t$要把自己的缩放给放在一个变量里，对$W_t$的缩放放在另外的变量里，恰好这两个变量的缩放
有一个公共的部分$\beta_t$所以就变成$(A_t + \alpha_t W_t ) / \beta_t$了，具体还要看看Wei Xu的paper。
