---
layout: post
title: "Elements of Statistical Learning笔记第2章"
date: 2014-06-15 12:01:04 -0400
comments: true
categories: MachineLearning Notes
---

ESL的笔记和正文提及的习题。这章主要介绍了supervised learning的基本概念和以后各章的摘要。

<!--more-->

## 术语 ##

在一个learning过程中涉及到的概念有inputs和outputs。inputs又叫做
predictors，independent variables，features。outputs又叫做responses，
dependent variables。

根据变量是quantitative还是qualitative来区分learning是regression还是
classification。还有第3种变量ordered categorical，例如small，medium，
large之类的。quantitative变量当然用实数表示。qualitative变量可以用
one-hot coding表示。
   
ESL比较好的一点是对用到的数学符号有准确的定义，而不像一些machine
learning的paper，实数和随机变量符号不分。有时候会让人看不懂。输入随机变
量用大写斜体$X$表示。如果$X$是向量，其中第j个元素用$X_j$表示。
qualitative变量$G$。第i个$X$的观测值不论是$X$是标量还是向量，都用
小写的$x_i$表示。N个观测值的一个集合用大写粗体的$\mathbf{X}$。向量
一般不粗体。有时候为了表示第$j$随机变量的N个观测值，也就是矩阵$X$的一
列的时候才会粗体$\mathbf{x}_j$。矩阵的一行也就是第$i$个随机变量观测值
$x_i^T$。

## LS和kNN ##
最简单的learning方法是least model和kNN。传说很多方法都是从这两个货改过来的。

$$ \hat{Y} = X^T\hat{\beta} $$

估计$Y$值。
$\hat{Y}$可以是变量也可以是向量。用

$$
\text{RSS}(\beta) = (\mathbf{y} - \mathbf{X}\beta)^T(\mathbf{y} - \mathbf{X}\beta)
$$

作为loss function去求$\hat{\beta}$。

$$
\hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1}X^T\mathbf{y}
$$

linear model用在classification问题上，就以$$0.5$$为界。

kNN的用

$$ \hat{Y}(x) = \frac{1}{k}\sum_{x_i \in N_k(x)} y_i $$

来估计$Y$值。看上去KNN只有一个参数$k$其实不然。因为kNN估计的是一个
neighborhood的里数据的mean，有多少个neighborhood就有多少个参数
$\frac{N}{k}$。kNN的有效参数个数一般都大于linear model的参数个数$p$。这
也印证了后面的bias-variance分解的结论：linear model是high bias，low
variance；kNN是low bias，high variance。

## Statistical Decision Theory ##
Statistical Learning Theory给出了一个学习框架。linear model和kNN可以自
然的从中推出。[Learning from Data](http://work.caltech.edu/telecourse.html)一书对这部分内容有非常通俗易懂的解释。
用统计学习理论回答了几个问题。什么是学习（learning）。机器能学习么？如
何学习？如何学得好。涉及了VC维，如何用Traing误差估计泛化误差，Cross
Validation是如何影响估计的和trade-off。[Foundations of Machine Learning](http://www.cs.nyu.edu/~mohri/mlbook/)
一书主要也是讲这个更加理论话一点。LFD研究最简单的二分类问题，loss
function也是最理想的分类损失函数。重要结论有

$$ \mathbb{P}[|
\text{E}_{\text{in}}(g) - \text{E}_{\text{out}}(g)| > \epsilon] \le 4
m_{\mathcal{H}}(2N)e^{-\frac{1}{8}\epsilon^2N} $$

简单点

$$E_{\text{out}} \le E_{\text{in}} + \Omega $$

意思是说in-sample和
out-sample的误差也就是泛化误差不会超过一个由VC维，数据量$N$和误差
$\epsilon$决定的界。看上去很美，其实大部分理论界都尼玛太松了，实际用
来估计误差大误。所以LFD最后介绍了Cross Validation并重新估计了泛化误差。
里面有对Validation的数据量的trade-off。ESL和LFD都有Bias-Variance分解内
容大同小异。[Machine Learning From A Probabilistic Perspective](http://www.cs.ubc.ca/~murphyk/MLbook/)也不错。公
式细节都很详细，就是钻研MLAPP容易陷入细枝末节，insights没有ESL那么多。
外功MLAPP，内功LFD和ESL。

ESL定义EPE。奇怪为啥ESL不把EPE的全称写出来，多年以前看的时候一直很纠结。
EPE叫expected prediction error。EPE需要一个loss function，$L(Y,f(X))$。
EPE可以用各种loss function，squared error loss，logistic loss，hinge loss。Tong Zhang有一篇paper对各种loss近似分类误差的分析。

如果用squared error loss，

$$
\text{EPE}(f) = \text{E}(Y-f(X))^2 = \int \left[y-f(x)\right]^2 \text{Pr}(dx, dy)
$$

用条件概率公式

$$
\text{EPE}(f) = \text{E}_X\text{E}_{Y|X}\left(\left|Y-f(X)\right|^2|X\right)
$$

得

$$
f(x) = \text{E}(Y|X=x)
$$

kNN的近似

$$\hat{f}(x) = \text{Ave}(y_i | x_i \in N_k(x))$$

对这个做了两个近似，一是用平均取代期望，二是用领域取代了X，来近似Y和X的条件概率。
当$N$和$k$趋向于无穷大，$k/N$趋向于$0$。kNN的$\hat{f}$就逼近了用期望的$f(x)$。

linear regression也做了近似。一是用全局的线性函数近似$f(X)$，EPE得到

$$
\beta = \left[ \text{E}\left(XX^T\right)\right]^{-1}\text{E}\left(XY\right)
$$

linear regression的近似

$$
\beta = \left[ \left(\mathbf{X}^T\mathbf{X}\right)\right]^{-1}\mathbf{X}^T\mathbf{y}
$$

看到不同了没有？一个是随机变量一个是数据观测值。一个是求期望一个是求平均。可能很难看出哪里求了平均。
如果$X$只是一个随机变量的话，那么EPE是$\text{E}(X^2)$，linear regression是$N \times 1$大小的
观测值向量$x^Tx$。所以这里的转置符号稍微有点区别。

这里为啥squared loss会联系到kNN和linear regression？后面会讲因为隐式假定了数据的的高斯分布。
如果用绝对值的loss，kNN得到中值median。
   
分类问题稍微有点区别。EPE要对每一个$G$的取值要逐个考虑。

$$
\text{EPE} = \text{E}\left[ L(G, \hat{G}(X))\right]
$$

用条件概率公式

$$
\text{EPE} = \text{E}_X\sum_{k=1}^K L \left[ L(\mathcal{G}_k, \hat{G}(X))\right]\text{Pr}\left(\mathcal{G}_k|X\right)
$$

用0-1 loss function得到 $\hat{G}(x) = \arg \min_{g\in\mathcal{G}}\left[ 1- \text{Pr}\left( g|X=x\right)\right]$
这个选最大概率的类来作为估计的分类器叫Bayes classifier。Bayes classifier的误差叫做Bayes rate。

kNN用majority vote近似Bayes rate。Pattern Classification一书中有证明。
kNN的误差最多是Bayes rate误差的两倍。忘记咋证的了。dummy-variable的
linear regression也可以Bayes classifier。

## 维数灾难 ##

如果一个单位球是高维的，那么球的“质量”绝大多数分布在球最外层的壳上。这个时候要是在想取原点的最近邻的话就
差不多要跑到球表面上去了。

Bias和Variance在不同的情况下对总的error贡献是不同的。以kNN为例。在对称函数的$f$中搞kNN，随着维数的增加，
bias增加，因为近邻会越来越趋向于1或者-1，函数值都趋近于0。但是因为对称的，variance不怎么增加，
基本上都是0左右。而另外一个用指数函数做的kNN，bias不怎么增加。因为最近邻随着维数的增加趋向于两端，
但是函数值是一大一小基本抵消。variance增加不少，因为有时候最近邻函数值很大（趋向于1），有时候很小（趋向于-1）。

以linear model为例子。

$$
\begin{align}
\text{EPE}(x_0) &= \text{Var}(y_0|x_0) + \text{Var}_{\mathcal{T}}(\hat{y}_0) + \text{Bias}^2(\hat{y}_0) \\
&= \sigma^2 + E_{\mathcal{T}}x_0^T\left(X^TX \right)^{-1}x_0\sigma^2 + 0^2
\end{align}
$$

这个公式在其他书中都有。trick无非是加一项减一项 $\text{E}_{\mathcal{T}}\hat{y}_0$。当$N$趋向于无穷大，$\text{E}(X) = 0$
的时候，用trace trick就$\text{trace}(x^TAx) = \text{trace}(Axx^T)$得到简化公式$\sigma^2(p/N)+\sigma^2$。

## Statistical Models ##

additive model假设那些$Y$偏离$f(X)$的部分都可以用独立同分布的$\epsilon$表示。这也是quantitative变量的
EPE中使用squared errors的原因。但是在qualitative变量中一般就不大可行。因为$Pr(G|X)$是条件概率。

learning问题也可以看做是函数近似。通常是在一个以 $\theta$为参数的函数family里面找出一个能最好的近似数据。
least squares其实是最大似然估计maximum likelihood estimation下用高斯假设的特例。

$$
L(\theta) = \sum_{i=1}^N \log \text{Pr}_\theta(y_i)
$$

其中 $Pr (Y\|X,\theta) = N(f_{\theta} (X), \sigma^2)$。

如果把最大似然估计用在qualitative变量做分类上，

$$
L(\theta) = \sum_{i=1}^N \log p_{g_i,\theta}(x_i)
$$

就得到了cross-entropy。如果对每一类的数据假设高斯分布的话，就是logistic regression了。所以logistic regression
也可以从Linear discriminant analysis推导出来。LDA假设数据是同Covariance的高斯分布，
用generative的方法，先写出高斯分布$P(X|Y)$，再用贝叶斯公式写出 $P(Y|X)$ ，MLAPP Page 102，假设同$\Sigma$。
其实就相当于直接假设$P(Y|X)$是sigmoid函数。

## Structured Regression Models ##
单单用RSS作为选择函数$f$的标准是不够的，因为所有使得在已知数据点上的误差为0的函数都使得RSS为0。所以要在一个比较小
函数集合内找RSS。这类方法叫做complexity restriction。大多数方法是在input的邻域上有一些简单的结构，smooth啦，liear啦，
多项式啦。成功的算法大多如此。对于高维问题，也有巧妙的metric的定义。
   
## Restrictions ##
对RSS增加一项smooth惩罚项$\int[f^{\prime\prime}(x)]$。projection pursuit regression也是在寻找参数的时候避免使得函数不smooth。

Kernel方法对一个邻域里的数据点赋予权重。这里对用kernel的RSS有了一个统一的解释。

$$
\text{RSS}(f_\theta, x_0) = \sum_{i=1}^N K_\lambda(x_0, x_i)(y_i, f_\theta(x_i))^2
$$

这里不像上面的RSS，这里的RSS还有个参数 $x_0$，所以基本的kernel methods像kNN一样特别得“local”。书中给出了两个特例。

Basis functions是用了一个基函数的集合。Neural Networks属于这一类。
   
## Model Selection和Bias-Variance trade-off ##
kNN regression的bias-variance分解。kNN regression的函数

$$
\hat{f}_k(x_0) = \frac{1}{k}\sum_{l=1}^k(f(x_l) + \epsilon_l)
$$

Bias非常直接。算Variance的时候要注意每一个$ x_l $都有不同的 $ \epsilon_l $。

$$
\text{Var}_{\mathcal{T}}\left( \hat{f}_k(x_0)\right) = \text{E}\left[ (\frac{1}{k}\sum_{l=1}^k \epsilon_l )^2\right]
$$

接下去因为$\epsilon_l$是独立同分布的。所以只有每个$\epsilon_l$的方差留了下来。最终得到书中的结果。
Bias和Variance之间的trade-off说的是model complexity的增加Bias减少了，Variance增加了。

## 练习
### 习题2.3
求从原点到最近邻的距离的中位数的公式。

$$ d(p, N) = \left( 1 - \frac{1}{2}^{1/N}\right)^{1/p} $$

因为数据点是均匀分布在$p$维单位球里。若原点的最近邻到原点的距离
$X$也是一个随机变量。那么其他所有的点都要落在$X$之外。
所以

$$ P(X = x)  = (1 - x^p)^N $$

根据median的定义$P(X\le m) \ge 0.5$, $P(X \ge m) \ge 0.5$。令上式右边等于0.5，解出
$x$就得到书中的公式了。

   
## 其他笔记
+ [demonstrate的笔记](http://remonstrate.wordpress.com/2011/10/24/%25E9%2587%258D%25E8%25AF%25BB-esl%25EF%25BC%2588%25E4%25B8%2580%25EF%25BC%2589/)
+ [落园的笔记](http://www.loyhome.com/%25E2%2589%25AA%25E7%25BB%259F%25E8%25AE%25A1%25E5%25AD%25A6%25E4%25B9%25A0%25E7%25B2%25BE%25E8%25A6%2581the-elements-of-statistical-learning%25E2%2589%25AB%25E8%25AF%25BE%25E5%25A0%2582%25E7%25AC%2594%25E8%25AE%25B0%25EF%25BC%2588%25E4%25BA%258C%25EF%25BC%2589/)
