#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:43
# @Author  : Tianyu Liu

import tensorflow as tf
import time
import numpy as np


class DataLoader(object):
    def __init__(self, limits, is_tseq2seq=True):
        if is_tseq2seq:  # template based seq2seq  S + sep + P
            self.train_data_path = ['../data/train/enc.id', '../data/train/dec.temp.id']
            self.test_data_path = ['../data/test/enc.id', '../data/test/dec.temp.id']
        else:  # vanilla seq2seq
            self.train_data_path = ['../data/train/enc.id', '../data/train/dec.full.id']
            self.test_data_path = ['../data/test/enc.id', '../data/test/dec.full.id']

        self.limits = limits
        start_time = time.time()

        self.train_set = self.load_data(self.train_data_path)
        self.test_set = self.load_data(self.test_data_path)

        print ('Reading datasets comsumes %.3f seconds' % (time.time() - start_time))

    def load_data(self, path):
        enc_path, dec_path = path
        encs = open(enc_path, 'r').read().strip().split('\n')
        decs = open(dec_path, 'r').read().strip().split('\n')
        if self.limits > 0:
            encs = encs[:self.limits]
            decs = decs[:self.limits]
        print(encs[0].strip().split(' '))
        encs = [list(map(int, enc.strip().split(' '))) for enc in encs]
        decs = [list(map(int, dec.strip().split(' '))) for dec in decs]
        return encs, decs

    def batch_iter(self, data, batch_size, shuffle):
        encs, decs = data
        data_size = len(encs)
        num_batches = int(data_size / batch_size) if data_size % batch_size == 0 \
            else int(data_size / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            encs = np.array(encs)[shuffle_indices]
            decs = np.array(decs)[shuffle_indices]

        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            max_enc_len = max([len(sample) for sample in encs[start_index:end_index]])
            max_dec_len = max([len(sample) for sample in decs[start_index:end_index]])
            batch_data = {'enc_in':[], 'enc_len':[],
                          'dec_in':[], 'dec_len':[], 'dec_out':[]}
            for enc, dec in zip(encs[start_index:end_index], decs[start_index:end_index]):
                enc_len = len(enc)
                dec_len = len(dec)
                gold = dec + [2] + [0] * (max_dec_len - dec_len)
                summary = dec + [0] * (max_dec_len - dec_len)
                text = enc + [0] * (max_enc_len - enc_len)
                batch_data['enc_in'].append(text)
                batch_data['enc_len'].append(enc_len)
                batch_data['dec_in'].append(summary)
                batch_data['dec_len'].append(dec_len)
                batch_data['dec_out'].append(gold)
            yield batch_data
