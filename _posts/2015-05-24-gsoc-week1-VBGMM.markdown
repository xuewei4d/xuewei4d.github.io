---
layout: post
title: "GSoC Week 1 VBGMM"
date: 2015-05-24T15:23:39-04:00
comments: true
categories: gsoc
---

# VBGMM
This week, I studied variational inference described in Chapter 10 of Pattern Recognition and Machine Learning (PRML) on GMM model. 
I derived the updating functions of VBGMM with "full" type covariance matrix. There are so many equations. Download the [file](https://www.dropbox.com/s/8hlbb7dlwllwcry/VBGMM.pdf?dl=0) from Dropbox. Currently, I have done
the updating functions with other three covariance matrix "tied", "diag" and "sphere", but I have not typed into the latex file yet.

I also studied the adjustment of GMM API. The discussion on issue [#2473](https://github.com/scikit-learn/scikit-learn/issues/2473), [#4062](https://github.com/scikit-learn/scikit-learn/issues/4062) points out the inconsistency on ``score_ sample``, ``score``. So I changed and made a new API interface of some functions in the [ipython notebook](http://nbviewer.ipython.org/gist/xuewei4d/de5492d0320eed561b78/GMM_API.ipynb?flush_cache=true). 

