---
layout: post
title: "GSoC Week 8, 9 and Progress Report 2"
date: 2015-07-27T10:59:05-04:00
comments: true
categories: gsoc
---

## Week 8 and 9
In the week 8 and 9, I implemented ```DirichletProcessGaussianMixture```.  But its behavior looks similar to ```BayesianGaussianMixture```. Both of them can infer the best number of components. ```DirichletProcessGaussianMixture``` took a slightly more iteration than ```BayesianGaussianMixture``` to converge on Old-faith data set, around 60 iterations.

If we solve Dirichlet Process Mixture by Gibbs sampling, we don't need to specify the 
truncated level T. Only the concentration parameter  $\alpha$ is enough. In the other hand, with variational inference, we still need to specify the maximal possible number of components, i.e., the truncated level.

At the first, the lower bound of ```DirichletProcessGaussianMixture``` seems a little strange. It is not always going up. When some clusters disappear, it goes down a little bit, then go up straight. I think it is because the estimation of the parameters is ill-posed when these clusters have data samples less than the number of features. I did the math derivation of Dirichlet process mixture models again, and found it was a bug on the coding of a very long equation.

I also finished the code of ```BayesianGaussianMixture``` for 'tied', 'diag' and 'spherical' precision.

My mentor pointed out the style problem in my code and docstrings. I knew PEP8 convention, but got no idea where was also a convention for docstring, PEP257. It took me a lot of time to fix the style problem.

## Progress report 2
During the last 5 weeks (since the progress report 1), I finished the 

1. ```GaussianMixutre``` with four kinds of covariance
2. Most test cases of ```GaussianMixutre```
3. ```BayesianGaussianMixture``` with four kinds of covariance
4. ```DirichletProcessGaussianMixture```

Although I spent some time on some unsuccessful attempts, such as decoupling out observation models and hidden models as mixin classes, double checking DP equations, I did finished the most essential part of my project and did some visualization. In the following 4 weeks, I will finish all the test cases for ```BayesianGaussianMixture``` and ```DirichletProcessGaussianMixture```, and did some optional tasks, such as different covariance estimators and incremental GMM.