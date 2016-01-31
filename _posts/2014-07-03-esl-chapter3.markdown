---
layout: post
title: "Elements of Statistical Learning笔记第3章"
date: 2014-07-03 13:34:28 -0400
comments: true
categories: Notes MachineLearning
---

ESL的笔记和正文提及的习题。这章主要介绍了linear regression。

<!--more-->

## Linear Regression Models and Least Squares ##

$$ f(X) = \beta_0 + \sum_{j=1}^pX_j\beta_j $$

$X$可以是quantitative；可以是quantitative的各种变换后的值，取对数，开方。取对数很常见；可以是基展开的；可以是
qualitative变量的coding；可以是多个变量的函数。

最常用的是以least square最小二乘作为优化函数。中文名最小二乘对我来说真是好陌生啊。我觉得在学术文章里是中英夹杂是
用来提高效率的。

$$ \text{RSS}(\beta) = \sum_{i=1}^N(y_i - \beta_0 - \sum_{j=1}^px_{ij}\beta_j) $$

least square隐含假设是i.i.d. 把$\mathbf{X}$表示用全1的列向量扩展的观察数据矩阵，对上面的式子求导得到看到过无数遍的
closed form，面试的时候被问过。

$$ \hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

对$\hat{\mathbf{y}}$的估计就是用下面的矩阵$\mathbf{H}$

$$\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

直接乘以$\mathbf{y}$得到。所以这个矩阵$\mathbf{H}$也就hat，这个也叫投影矩阵。自己乘自己还是自己。
落园的笔记的还提到了一个消灭矩阵$\mathbf{I}-\mathbf{H}$，反正就是
为了计算误差方便的矩阵。

笔锋一转，几何乱入。把RSS用矩阵形式表达并且对$\mathbf{X}$按列分块。least square做的其实也是在由
$\mathbf{X}$的列向量张成的线性空间里找到一个$\mathbf{y}$的近似，使得残差最小。最优值也就是在残差向量垂直于
张成的线性空间的时候，也就是$\mathbf{y}$作投影的时候得到。

毕竟是频率学派，对$\hat{\beta}$的分析和假设检验出场。因为

$$\hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{X}\beta + \boldsymbol{\epsilon})$$

所以$\text{E}(\hat{\beta}) = \beta$。这里要注意的是$\boldsymbol{\epsilon}$是一个由$N$个不同的$\epsilon$组成的向量，每个$\epsilon$服从i.i.d分布。这里一直误以为是对每个元素求方差，其实ESL里面不区分方差Variance和协方差Covariance。这里Variance就是
Covariance。怪不得之前算方差的时候这么别扭，T.T。而且ESL里面不细分点乘和矩阵乘，要自行辨认哪个是标量哪个是点乘。
陈希孺的书中有几个有用的公式，用ESL的符号表示就是，其中$\mathbf{A}$是常数矩阵

$$
\text{Var}\left[ \mathbf{A}X \right] = \mathbf{A} \text{Var} [ X ] \mathbf{A}^T
$$

按照这个公式和$\text{Var}[\boldsymbol{\epsilon}] = \sigma^2 \mathbf{I}$，最终得到

$$ \text{Var} [ \hat{\beta} ] = \text{E}\left[
(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \sigma^2  \mathbf{I}  \mathbf{X}
(\mathbf{X}^T\mathbf{X})^{-1} \right] = (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2
$$

所以$\hat{\beta}$服从正态分布。

$$
\hat{\beta} \sim N(\beta, (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2)
$$



以下结论和证明都可以在陈希孺的近代回归分析中找到。因为$\sigma^2$是未知的，要通过

$$
\hat{\sigma}^2 = \frac{1}{N-p-1}\sum_{i=1}^N(y_i - \hat{y}_i)^2
$$

来对$\sigma^2$无偏估计。这个证明匆匆看了下，关键是用到$\mathbf{X}^T\mathbf{X}$的秩最多$p+1$。另外

$$
(N-p-1)\hat{\sigma}^2 \sim \sigma^2\chi_{N-P-1}^2
$$

这个证明关键是把RSS表示成二次型$\text{RSS} = \mathbf{y}(\mathbf{I} - \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T)\mathbf{y}$，和多个正态分布相加的形式。还有

$$
z_j = \frac{\hat{\beta}_j}{\hat{\sigma}^2\sqrt{v_j}} \sim t_{N-p-1}
$$

这个本身就是$t$分布的定义，具体的概率密度函数pdf是后来推导出来的，巨复杂。第1和第2个证明还好。当样本数量
足够大的时候$z_j$近似服从正态分布了。有了这个分布之后，我们就可以做假设检验看某个随机变量是否显著。也可以用
F分布对一个随机变量的集合做假设检验。这里还是就记一个结论吧。线性回归在统计里水应该还是蛮深的，可以讲一本书。和陈希孺的
近代线性回归和Seber的线性回归分析书比较，ESL算讲得少了，但还算比较深刻。MLAPP第7章只有几个公式，没有多少篇幅
具体讨论。比如没有QR分解和施密特正交化的关系。

## 高斯-马尔科夫定理 GM Theorem
证明见后文。least square是所有无偏估计中方差最小的。但是learning的最终问题是泛化误差或者叫做预测误差。任何一本
Machine Learning书中都会说到泛化误差可以分为噪声的方差 + bias^2 + variance。所以有的估计稍微提高一点bias但是
可以大幅降低variance，得到更好的泛化误差。最近的ICML有对MCMC的改进也是这个思路。

## Multiple Regression from Simple Univariate Regression
这又是一个新的观点从中引出了QR分解和后面的特征选择。这个idea是，如果$\mathbf{X}$的每一列都是互相正交的话，那么
least square linear regression做起来非常简单，就是$<\mathbf{x}_j, \mathbf{y}>/<\mathbf{x}_j, \mathbf{x}_j>$。
所以只要对$\mathbf{X}$的列向量正交化就好了。单变量回归对于有截距intercept怎么办呢？从中推导出的公式相当于
$\beta_0$等于$\bar{\mathbf{x}}$，再把$\mathbf{x}$在前面的残差向量$\mathbf{z}$做regression。

那么一般形式就是在用全1向量扩展的$\mathbf{X}$上做schmidt正交化过程。正交化的结果就是一个正交矩阵和一个上三角矩阵。

$$
\mathbf{X} = \mathbf{Z}\boldsymbol{\Gamma} = \mathbf{Q}\mathbf{R}
$$

在标准化一下就是QR分解。用标准正交矩阵性质带入公式，得到

$$
\mathbf{R}\hat{\beta} = \mathbf{Q}^T\mathbf{y}
$$

这个线性方程组很容易算，back substitution。

## Subset Selection
这里所有的特征选择是两步走的，先把一个feature选进来，拟合一下看一下好不好，不好下一个。Best-Subset Selection，Forward，
Backward-stepwise，Forward-Stagewise。

## Shrinkage Methods
Ridge regression就是加了一种bias。

$$
\min  \left\{ \sum_{i=1}^N (y_i - \beta_0 - \sum_{j=1}^p x_{ij} \beta_j)^2 +
\lambda \sum_{j=1}^p \beta_j^2  \right\}
$$

这种优化在Convex Optimization一书中叫做Scalarization。最优点叫做Pareto optimal。注意这里没有$\beta_0$的惩罚项。因为我们
可以把$\mathbf{Y}$都加上或处以某个相同的数，那么回归得到的系数除$\beta_0$以外应该不变才对。如果加上了$\beta_0$的惩罚
之后，其他系数就没有这种不变性了。

书中提到直接优化目标函数可以分为两步走。第一步把$\mathbf{X}$去中心化，$\beta_0=\bar{y}$。第二部用没有intercept的
ridge regression。这也是练习3.5，证明见后文。

解析解也是看了无数遍的

$$
\hat{\beta} = (\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
$$

还有可以联系贝叶斯后验概率。SVD分解又提供了另外一个角度。

$$
\mathbf{X}\hat{\beta} = \mathbf{U}\mathbf{U}^T\mathbf{y}
$$

变成了

$$
\mathbf{X}\hat{\beta} = \sum_{j=1}^p \mathbf{u}_j\frac{d_j^2}{d_j^2 + \lambda}\mathbf{u}_j^T\mathbf{y}
$$

ridge regression就是把$\mathbf{u}$缩放了一下。这种缩放有什么好处呢？联系PCA。

如果$d_j$比较小的话，


## 练习

### 习题3.3

$\hat{\beta}$的期望就是$\beta$。书中的
期望对$\beta$还乘了个向量$a$。之前都是自己死推无比繁琐。有了上面的方差
公式就方便多了，落园博客上的证明就冗长了。之于为什么要乘以一个向量，我
估计是因为最终比较的方差的大小。如果不乘的话，比较的是两个协方差矩阵，
怎么比么。写博客的一个好处是以前推导中疏忽的地方会纠正过来。这个是书中
习题。这里$\mathbf{c}$和$\mathbf{y}$粗体的原因是因为有$N$个元素。
因为$\text{E}[\mathbf{c}^T\mathbf{y}] = a^T\beta$，则把$\mathbf{y}$代换掉，
得到$a = \mathbf{X}^T\mathbf{c}$。

$$
\text{Var}[\mathbf{c}^T\mathbf{y}]
= \sigma^2 \mathbf{c}^T \mathbf{c} \label{first}
$$

以及

$$ \text{Var}[a^T\hat{\beta}]
= \sigma^2 \mathbf{c}^T \mathbf{X} (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T\mathbf{c} \label{second}
$$

所以只要证明第一个式子的二次型中的矩阵$\mathbf{I}$减去第二个式子的是半正定的就好了。用SVD分解$\mathbf{X}$。

$$ \mathbf{X} (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^T  = \mathbf{U}\Gamma\mathbf{U}^T$$。

其中$\mathbf{U}$是$N \times N$的矩阵。那么$\mathbf{I} - \mathbf{U}\Gamma\mathbf{U}^T$必然是半正定的，因为
用$\Gamma$只有左上角的那个小块矩阵是单位矩阵。


### 习题3.5
网上有些习题是把书中公式3.41的$x_{ij}$变成$x_{ij} - \bar{x}_j + \bar{x}_j$，然后比较

$$
\arg \min \left\{\sum_{i=1}^N [ y_i - \beta_0^c - \sum_{j=1}^p(x_{ij} - \bar{x}_j)\beta_j^c ]^2
+ \lambda \sum_{j=1}^p {\beta_j^c}^2\right\}
$$

得到$$\hat{\beta}^c_0 = \hat{\beta}_0 - \sum^{p}_{j=1} \bar{x}_j\hat{\beta}_j $$。

我觉得这里很有问题。因为一般用这种比较方法的时候都是两个公式是恒等式。而这里是求最小值。如果在从上面得到第二个式子做最小值的时候，必须考虑
$$\hat{\beta}_0^c$$
和其他$$\hat{\beta}_j^c$$之间的约束$$\hat{\beta}_0^c = \hat{\beta}_0 - \sum_{j=1}^p\bar{x}_j\hat{\beta}_j$$。所以这里导出的第二个优化函数多了一个约束。如果得到了$\hat{\beta}^c$之后还要验证转换得到的
$\hat{\beta}$是否是使得原优化最小。

我的做法是两个优化公式直接求$\beta$和$\beta^c$然后比较是否相等，虽然麻烦了点。引入一个叫做centering matrix

$$\mathbf{C} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T$$

左乘这个矩阵相当于做中心化。$\mathbf{C}$是对称的。性质有$\mathbf{1}^T\mathbf{C} = \mathbf{0}^T$，
$\mathbf{C}^T\mathbf{C} = \mathbf{C}$。做1遍以上去中心化就相当于只做了1遍。

令$\beta = [\beta_0, \tilde{\beta}]$，其中$\tilde{\beta}$是$\beta$从1到p的向量。
去中心化后的优化函数写成矩阵的形式是

$$
\min \|\mathbf{y} - [\mathbf{1}, \mathbf{C}\mathbf{X}]\beta^c \|_2^2 + \| \tilde{\beta}^c \|_2^2
$$

导数等于0。用分块矩阵写出来是

$$
\begin{bmatrix}
\mathbf{1}^T\mathbf{y} \\ \mathbf{X}^T\mathbf{C}^T\mathbf{y} 
\end{bmatrix}
=
\begin{bmatrix}
n & \mathbf{1}^T\mathbf{C}\mathbf{X} \\ \mathbf{X}^T\mathbf{C}^T\mathbf{1} & \mathbf{X}^T\mathbf{C}^T\mathbf{C}\mathbf{X} + \mathbf{I}
\end{bmatrix}
\begin{bmatrix}
\beta^c_0 \\
\tilde{\beta}^c
\end{bmatrix}
$$

把$\tilde{\beta}^c$提出来之后得到

$$
\mathbf{X}^T\mathbf{C}^T\mathbf{y} = (\mathbf{X}^T\mathbf{C}\mathbf{X} +\mathbf{I})\tilde{\beta}^c
$$

另外一方面，原始的优化函数写成矩阵形式并求导等于0之后

$$
\begin{bmatrix}
\mathbf{1}^T\mathbf{y} \\
\mathbf{X}^T\mathbf{y}
\end{bmatrix}
=
\begin{bmatrix}
n & \mathbf{1}^T\mathbf{X} \\
\mathbf{X}^T\mathbf{1} & \mathbf{X^T}\mathbf{X}
\end{bmatrix}
\begin{bmatrix}
\beta_0 \\
\tilde{\beta}
\end{bmatrix}
$$

解个2元一次方程组。把下面

$$
\beta_0 = \frac{1}{n}\left[\mathbf{1}^T\mathbf{y} - \mathbf{1}^T\mathbf{X}\tilde{\beta}\right]
$$

代入

$$
\mathbf{X}^T\mathbf{1}\beta_0 + (\mathbf{X}^T\mathbf{X} + \mathbf{I})\tilde{\beta} = \mathbf{X}^T\mathbf{y}
$$

整理并用$\mathbf{C}$的定义，$\tilde{\beta}$就得到了同$\tilde{\beta}^c$一样的样子了。

所以这道题无非说的是，最小二乘线性回归其实可以理解成，先做了去中心化，得到$\beta_0$，把他从$\mathbf{y}$里减掉，
剩下的是在做无截距的最小二乘。这样公式3.43中的$\beta$可以写的没联系中的那么别扭了。MLAPP第226页也有但是没提这一点
可能读到的时候会觉得奇怪吧。

### 习题3.8
有了练习3.5，3.8非常相似，简单得证。
故事说的是
