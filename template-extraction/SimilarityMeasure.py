#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-6-30 下午11:25
# @Author  : Tianyu Liu

#encoding=utf-8


import gc

from gensim import corpora, models, similarities
from collections import defaultdict
import copy

class SentenceSimilarity():

    def __init__(self, target):
        self.target = target
        self.length = len(self.target)
        pairs = self.combine(range(self.length), 2)
        self.group1, self.group2 = [], []
        for pair in pairs:
            id1, id2 = pair
            if id1 == id2:
                continue
            else:
                self.group1.append(self.target[id1])
                self.group2.append(self.target[id2])


    def combine(self, lst, l):
        result = []
        tmp = [0] * l
        length = len(lst)

        def next_num(li=0, ni=0):
            if ni == l:
                result.append(copy.copy(tmp))
                return
            for lj in range(li, length):
                tmp[ni] = lst[lj]
                next_num(lj + 1, ni + 1)

        next_num()
        return result

    # 构建其他复杂模型前需要的简单模型
    def simple_model(self, min_frequency = 0):
        # 删除低频词
        frequency = defaultdict(int)
        for text in self.group1:
            for token in text:
                frequency[token] += 1

        self.texts = [[token for token in text if frequency[token] > min_frequency] for text in self.group1]
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus_simple = [self.dictionary.doc2bow(text) for text in self.texts]
        # self.corpus_simple.append(self.corpus_simple[0])
        # print(self.corpus_simple[0])
        # print(self.corpus_simple[1])

    # tfidf模型
    def TfidfModel(self):
        self.simple_model()

        # 转换模型
        self.model = models.TfidfModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]
        # 创建相似度矩阵
        self.index = similarities.MatrixSimilarity(self.corpus)

    # lsi模型
    def LsiModel(self):
        self.simple_model()

        # 转换模型
        self.model = models.LsiModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]

        # 创建相似度矩阵
        self.index = similarities.MatrixSimilarity(self.corpus)

    #lda模型
    def LdaModel(self):
        self.simple_model()

        # 转换模型
        self.model = models.LdaModel(self.corpus_simple)
        self.corpus = self.model[self.corpus_simple]

        # 创建相似度矩阵
        self.index = similarities.MatrixSimilarity(self.corpus)

    # 求最相似的句子
    def similarity(self):
        total = 0
        # print(len(self.group2))
        for idx, sentence in enumerate(self.group2):
            # print(idx)
            sentence_vec = self.model[self.dictionary.doc2bow(sentence)]
            # sentence_vec = self.sentence2vec(sentence)
            sims = self.index[sentence_vec]
            # print(len(sims))
            score = sims[idx]
            # print(score)
            total += score
        total /= (idx + 1)
        return total
