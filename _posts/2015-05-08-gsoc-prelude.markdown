---
layout: post
title: "GSoC Prelude"
date: 2015-05-08T14:03:39-04:00
comments: true
categories: gsoc
---

It is fortunate that my proposal about Gaussian mixture model is accepted by
Google Summer of Code 2015. I am very grateful to scikit-learn, Python
Software Foundation and Google Summer of Code. As a PhD student studying in
Machine Learning and Data Mining, I frequently process various kinds of data
using Matlab, Python and scikit-learn. Scikit-learn is a powerful and easy-to-
use machine learning library for Python. Though I only have been using it for
about one year, I cannot leave it in my many projects now.

I first heard of GSoC in 2012, when my colleague
[*pluskid*](http://blog.pluskid.org/?p=865) participated in Shogun project.
The post he wrote about his experience is quite interesting and fun. Since I
missed GSoC 2014 because of too much course projects, I began to  read some
code of scikit-learn and learn git. Anyway, I really looking forward to a
wonderful journey this summer.

# Introduction

This summer, I focus on Gaussian mixture model and other two variances.
Compared with other two GSoC projects, my project looks a bit different, since
it is kind of fixing / refactoring rather than introducing new features. The
following text is from my proposal.

Gaussian mixture model (GMM) is a popular unsupervised clustering
method. GMM corresponds a linear combination of several Gaussian distributions
to represent the probability distribution of observations. In GMM, with the
prefix number of Gaussian component, a set of parameters should be estimated
to represent the distribution of the training data. It includes means,
covariances and the coefficients of the linear combination. Expectation
Maximization (EM) is usually used to find the maximum likelihood parameters of
the mixture model. In each iteration, E-step estimates the conditional
distribution of the latent variables.  M-step finds the model parameters that
maximize the likelihood.

In variational Bayesian Gaussian mixture model (VBGMM), M-step is generalized
into full Bayesian estimation, where the parameters are represented by the
posterior distribution, instead of only single value like in maximum-
likelihood estimation.

On the other hand, Dirichlet process Gaussian mixture model (DPGMM) allows a
mixture of infinite Gaussian distributions. It uses Dirichlet process as a
nonparametric prior of the distribution parameters, and the number of
components could vary according to the data. Therefore, one does not have to
preset the number of components ahead of time. The simplest way to infer DPGMM
is Monte-Carlo Markov chain (MCMC), but it is generally slow to converge. In
Blei's paper, truncated variational inference is proposed, which converges
faster than MCMC.

However, in scikit-learn, the implementation suffers from interface
incompatibility, incorrect model training and incompleteness of testing, which
prohibits the widely use of these models.

# Next 

In the rest of bonding time, I will continue reading the related papers. The
next post will be about mathematical derivation. Stay tuned. 
