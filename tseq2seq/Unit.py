#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:37
# @Author  : Tianyu Liu
from LstmUnit import LstmUnit
from OutputUnit import OutputUnit
from SeqUnit import SeqUnit
import os, shutil

class Vocab(object):
    """docstring for vocab"""
    def __init__(self):
        vocab = dict()
        vocab['UNK'] = 0
        vocab['PAD'] = 1
        vocab['START'] = 2
        vocab['SEP'] = 3
        vocab['(SUB)'] = 4
        cnt = 5
        with open("../data/vocab.txt", "r") as v:
            for line in v:
                word = line.strip()
                vocab[word] = cnt
                cnt += 1
        self._word2id = vocab
        self._id2word = {value: key for key, value in vocab.items()}

    def word2id(self, word):
        ans = self._word2id[word] if word in self._word2id else 0
        return ans

    def id2word(self, id):
        ans = self._id2word[int(id)]
        return ans

def to_word(pred_list, save_dir):
    v = Vocab()
    ss = open(save_dir + "test_summary.txt", "w+")
    for item in pred_list:
        ss.write(" ".join([v.id2word(int(id)) for id in item]) + '\n')


def copy_file(dst, src=os.getcwd()):
    files = os.listdir(src)
    for file in files:
        file_ext = file.split('.')[-1]
        if file_ext=='py':
            shutil.copy(os.path.join(src,file), dst)
