#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-6-15 上午12:15
# @Author  : Tianyu Liu
import re, os, shutil


def build_vocab():
    '''
    build vocabulary from training data
    '''
    ftrain = open('nlpcc-iccpol-2016.kbqa.training-data', 'r')
    vocab = open('data/vocab.txt', 'w+')
    fr2id = open('data/relation2id.txt', 'w+')
    word2id, id2word = {}, {}
    rel2id = []
    word2id['UNK'] = 0
    word2id['PAD'] = 1
    word2id['START'] = 2
    pattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])')
    cnt = 3
    for line in ftrain:
        if line.find('<q') == 0:  # question line
            qRaw = line[line.index('>') + 2:].strip()
        elif line.find('<t') == 0:  # triple line
            triple = line[line.index('>') + 2:]
            s = triple[:triple.index(' |||')].strip()  # topic word
            triNS = triple[triple.index(' |||') + 5:]
            p = triNS[:triNS.index(' |||')]  # predicate
            p, num = pattern.subn('', p)
            for sen in [qRaw, s, p]:
                for char in sen:
                    if char not in word2id:
                        word2id[char] = cnt
                        cnt += 1
            if p not in rel2id:
                rel2id.append(p)
        else:
            continue
    word2id = sorted(word2id.items(), key=lambda d:int(d[1]))
    for item in word2id[3:]:
        vocab.write(item[0]+'\n')
    for r in rel2id:
        fr2id.write(r + '\n')
    print("total number of vocabulary: %s" % str(len(word2id)))
    print("total number of predicates: %s" % str(len(rel2id)))
    vocab.close()
    fr2id.close()


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
        prefix = os.path.dirname(__file__)
        with open("data/vocab.txt", "r") as v:
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


def build_data(mode='train'):
    fread = open('nlpcc-iccpol-2016.kbqa.' + mode + 'ing-data', 'r')
    ftop = open('data/' + mode + '/topic.id', 'w+')
    fenc = open('data/' + mode + '/enc.id', 'w+')
    fdec = open('data/' + mode + '/dec.full.id', 'w+')
    tdec = open('data/' + mode + '/dec.temp.id', 'w+')
    fwenc = open('data/' + mode + '/enc.text', 'w+')
    fwdec = open('data/' + mode + '/dec.full.text', 'w+')
    v = Vocab()
    pattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])')
    cnt = 0
    for line in fread:
        if line.find('<q') == 0:  # question line
            qRaw = line[line.index('>') + 2:].strip()
            cnt += 1
        elif line.find('<t') == 0:  # triple line
            triple = line[line.index('>') + 2:]
            s = triple[:triple.index(' |||')].strip()  # topic word
            triNS = triple[triple.index(' |||') + 5:]
            p = triNS[:triNS.index(' |||')]  # predicate
            p, num = pattern.subn('', p)

            # subject entity in the question
            topic = [str(v.word2id(char)) for char in s]

            # raw text for question (for decoding)
            dec_text = [char for char in qRaw]
            qfdec = [str(v.word2id(char)) for char in qRaw]

            # replace topic words in raw question with (SUB)
            qRplc = qRaw
            if qRaw.find(s) != -1:
                qRplc = qRaw.replace(s, '^', 1)
            qtdec = [str(v.word2id('(SUB)')) if char == '^' else str(v.word2id(char)) for char in qRplc]

            # topic words + '|||' + predicate  for encoding
            enc_text = [char for char in s] + ['|||'] + [char for char in p]
            qenc = [str(v.word2id(char)) for char in s] + [str(v.word2id('SEP'))]
            qenc += [str(v.word2id(char)) for char in p]

            fdec.write(' '.join(qfdec) + '\n')
            tdec.write(' '.join(qtdec) + '\n')
            fenc.write(' '.join(qenc) + '\n')
            fwdec.write(' '.join(dec_text) + '\n')
            fwenc.write(' '.join(enc_text) + '\n')
            ftop.write(' '.join(topic) + '\n')
        else:
            continue
    print("number of (question,triple) pairs in " + mode + " set: " + str(cnt))


# split gold as separate files for ROUGE
# we replace the original text with ids because chinese characters will cause problems
# because the ROUGE script here use English wordnet
# For unk tokens, we replace unk token in gold set as -1 while replacing unk token in pred set as 0
def split_for_rouge():
    test = open('data/test/dec.full.id', 'r')
    gold_path = open('tseq2seq/reference.txt', 'w+')  # for BLEU metric in tseq2seq model
    split_test_path = 'run/evaluation/gold/question_'
    k = 0
    for line in test:
        items = line.strip().split()
        new_item = []
        for item in items:
            if item == '0':
                new_item.append('-1')
            else:
                new_item.append(item)
        with open(split_test_path + str(k), 'w+') as sw:
            sw.write(' '.join(new_item) + '\n')
        gold_path.write(' '.join(new_item) + '\n')
        k += 1


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

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/train/'):
        os.mkdir('data/train/')
    if not os.path.exists('data/test/'):
        os.mkdir('data/test/')
    if not os.path.exists('run'):
        os.mkdir('run')
    if not os.path.exists('run/evaluation/'):
        os.mkdir('run/evaluation/')
    if not os.path.exists('run/res/'):
        os.mkdir('run/res/')
    if not os.path.exists('run/evaluation/gold/'):
        os.mkdir('run/evaluation/gold/')
    if not os.path.exists('run/evaluation/gold_temp/'):
        os.mkdir('run/evaluation/gold_temp/')
    if not os.path.exists('run/evaluation/pred_temp/'):
        os.mkdir('run/evaluation/pred_temp/')
    if not os.path.exists('run/evaluation/pred_s2s/'):
        os.mkdir('run/evaluation/pred_s2s/')
    if not os.path.exists('run/evaluation/pred_ts2s/'):
        os.mkdir('run/evaluation/pred_ts2s/')
    print("building vocab for training and testing ...")
    build_vocab()
    print("done ...")
    print("building train and test set ...")
    build_data('train')
    build_data('test')
    print("done ...")
    split_for_rouge()


