---
title: Notes on A Neural Attention Model for Abstractive Sentence Summarization
layout: post
date: 2016-03-11  -0400
comments: true
categories: Paper
---
Author Alexander M. Rush, Sumit Chopra, Jason Weston

Novelty (A greatest impact, B novel method, C application and modification) C

Source EMNLP 2015

Comment
Abstractive Summarization with Bahdanau的machine translation attention。
模型大概是用一个score function来表示输入的句子单词X和输出的单词y的似然。输出句子中的下一个单词的概率一个关于当前已经生成的单词和输入单词x的函数。但是他们在encoding context的时候用了3中不同的方法，其中一个是用一个关于x和当前y的去生成分布于x的分布。在testing的时候用beam search生成最有可能的单词。但是看效果就感觉只是把修饰成分去掉了而已，有些还不对。

Dataset
DUC2003, DUC2004, Gigaword

Metric
ROUGE

Implementation
Torch

Time
4 days