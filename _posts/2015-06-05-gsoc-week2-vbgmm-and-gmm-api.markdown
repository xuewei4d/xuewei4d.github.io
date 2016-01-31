---
layout: post
title: "GSoC Week 2 VBGMM and GMM API"
date: 2015-06-05T15:00:30-04:00
comments: true
categories: gsoc
---

# VBGMM
I finally finish writing all derivations and equations for VBGMM with LaTex. 
**[ Download the derivation draft](https://www.dropbox.com/s/8hlbb7dlwllwcry/VBGMM.pdf?dl=0)**.
It is crazy to write equations of 12 pages in blog. So I wrote them in a traditional LaTex file.
Typing math equations is always pain. I have to be careful with ```mathbf```, ```boldsymbol```, subscripts.
It is boring and not cool to type ```\mathbf```, ```\boldsumbol```, ```\mu```, ```\Lambda``` again and again. 
There are 440 occurrences of ```boldsymbol``` :|. So I created several snippets in Sublime for typing these
LaTex commands.
I also learned some interesting advanced LaTex techniques.
There are so many extremely long equations have 9 or 10 terms, and each team is either ```frac``` or horrible $\sum$ or the productions of vectors.
Environments ```split```, ```align```, ```aligned```, ```empheq``` are very helpful to type those LaTex words. 

Well, they are not big deals. The most important thing is there is no derivations for VBGMM in the current sklearn docs.
We only found a [doc](http://scikit-learn.org/stable/modules/dp-derivation.html) about derivation for DPGMM.
Yes, I am done with VBGMM if there is no mistake after double-checking, and I am going to study DPGMM.
There is a little difference in the problem setting.

In the current doc,
$$
\begin{align}
\boldsymbol{\mu}_k & \sim \mathcal{N}(0, \mathbf{I}) \\
\boldsymbol{\Lambda}_k & \sim \mathcal{W}(\mathbf{I}, D)
\end{align}
$$
which is not the same as the setting in PRML
$$
\begin{align}
\boldsymbol{\mu}_k & \sim \mathcal{N}(\mathbf{m}_0, (\beta_0\boldsymbol{\Lambda}_k^{-1})) \\
\boldsymbol{\Lambda}_k & \sim \mathcal{W}(\mathbf{W}_0, \nu_0)
\end{align}
$$
I think the difference will make the final updating functions are different from the current implementations.

The trick about those derivation is 'completing the square', which is identify the second-order terms and one-order terms in the 
equations, and use the coefficients before these terms to 'make' the probability density function we want, then normalize it to absorb other constants in the equations.

# GMM API
After a stupid trying of deprecating old ```GMMM``` class, I created a new ```GaussianMixtureModel``` to keep the naming conventions,
and re-implement old ```GMM``` module inheriting from ```GaussianMixtureModel```. 
The new ```GaussianMixtureModel``` has reasonable ```score```, ```score_samples``` API which is coherent with other modules of sklearn.
The new ```DensityMixin``` class implements ```score``` and serves a mixin class for all current and future density estimators.
Mixing class technique is cool. I never heard this before I dig into the code base of sklearn.


# Next Week
I hope I could finish the derivations of DPGMM, and clean up GMM API.


