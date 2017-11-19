#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-11-13 下午10:29
# @Author  : Tianyu Liu


import re, os
import random
import sys
from PythonROUGE import PythonROUGE
from bleu import BLEU
from nltk.translate.bleu_score import corpus_bleu
from SimilarityMeasure import SentenceSimilarity


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

def get_rel2group():
    fi = open('../nlpcc-iccpol-2016.kbqa.testing-data', 'r')

    cnt = 0
    rel2group = {}
    for line in fi:
        if line.find('<q') == 0:  # question line
            continue
        elif line.find('<t') == 0:  # triple line
            triple = line[line.index('>') + 2:]
            triNS = triple[triple.index(' |||') + 5:]
            p = triNS[:triNS.index(' |||')]
            if p not in rel2group:
                rel2group[p] = [cnt]
            else:
                rel2group[p].append(cnt)
        else:
            continue
        cnt += 1
    with open('../data/relation2group.txt', 'w+') as wf:
        for key in rel2group:
            wf.write(' '.join([str(x) for x in rel2group[key]]) + '\n')


# extract templates from training set
def getAnswerPatten():
    fi = open('../nlpcc-iccpol-2016.kbqa.training-data', 'r')
    qRaw = ''
    pattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])')  # pattern to clean predicate, in order to be consistent with KB clean method

    APList = {}
    cnt = 0
    for line in fi:
        if line.find('<q') == 0:  # question line
            qRaw = line[line.index('>') + 2:].strip()
            continue
        elif line.find('<t') == 0:  # triple line
            triple = line[line.index('>') + 2:]
            s = triple[:triple.index(' |||')].strip()
            triNS = triple[triple.index(' |||') + 5:]
            p = triNS[:triNS.index(' |||')]
            p, num = pattern.subn('', p)
            if qRaw.find(s) != -1:
                qRaw = qRaw.replace(s, '(SUB)', 1)

            qRaw = qRaw.strip() + '\t|||\t' + p

            if qRaw in APList:
                APList[qRaw] += 1
            else:
                APList[qRaw] = 1
            cnt += 1
        else:
            continue
    fotxt = open('trainPattern.txt', 'w')
    for key in APList:
        fotxt.write(key + '\t' + str(APList[key]) + '\n')
    fotxt.close()


def pattern_baseline():
    v = Vocab()

    # files for evaluating BLEU
    pred_path, gold_path = 'candidate.txt', 'reference.txt'
    pred, gold = open(pred_path, 'w+'), open(gold_path, 'w')

    ftest = open('../nlpcc-iccpol-2016.kbqa.testing-data', 'r')

    # separate files for ROUGE
    # here we use different gold file from seq2seq because the extracted templates from training set
    # can't cover all the predicates from testing set
    gold_for_ROUGE = "../run/evaluation/gold_temp/question_"
    pred_for_ROUGE = "../run/evaluation/pred_temp/question_"

    # patterns extracted from training set
    trainAP = open('trainPattern.txt', 'r')
    rel_dic = {}
    for line in trainAP:
        line = line.strip()
        pattern, rel = line.split('\t')[0], line.split('\t')[-2]
        if rel not in rel_dic:
            rel_dic[rel] = [pattern]
        else:
            rel_dic[rel].append(pattern)

    pattern = re.compile(r'[·•\-\s]|(\[[0-9]*\])')
    cnt = 0
    gold_all, pred_all = [], []
    for line in ftest:
        if line.find('<q') == 0:  # question line
            qRaw = line[line.index('>') + 2:].strip()
            continue
        elif line.find('<t') == 0:  # triple line
            triple = line[line.index('>') + 2:]
            s = triple[:triple.index(' |||')].strip()  # topic word
            triNS = triple[triple.index(' |||') + 5:]
            p = triNS[:triNS.index(' |||')]  # predicate
            p, num = pattern.subn('', p)
            if p not in rel_dic:
                with open(pred_for_ROUGE+str(cnt), 'w+') as sw:
                    sw.write('\n')
                with open(gold_for_ROUGE+str(cnt), 'w+') as sw:
                    sw.write('\n')
                pred_all.append([])
                gold_all.append([])
            else:
                sp = random.sample(rel_dic[p],1)[0]
                sp = sp.replace('(SUB)', s)
                pred_list, gold_list = [], []
                for char in sp:
                    wid = v.word2id(char)
                    pred_list.append(str(wid))  # replace unk in pred list with 0
                pred_all.append(pred_list)
                pred.write(' '.join(pred_list) + '\n')
                with open(pred_for_ROUGE + str(cnt), 'w+') as sw:
                    sw.write(' '.join(pred_list) + '\n')
                for char in qRaw:
                    wid = v.word2id(char)
                    gold_list.append(str(-1 if wid == 0 else wid))  # replace unk in gold list with -1
                gold_all.append([gold_list])
                gold.write(' '.join(gold_list) + '\n')
                with open(gold_for_ROUGE + str(cnt), 'w+') as sw:
                    sw.write(' '.join(gold_list) + '\n')
            cnt += 1
        else:
            continue
    pred.close()
    gold.close()
    print("number of questions in test set: " + str(len(pred_all)))
    pred_set = [pred_for_ROUGE + str(i) for i in range(cnt)]
    gold_set = [[gold_for_ROUGE + str(i)] for i in range(cnt)]

    bleu = BLEU(pred_path, gold_path)
    print("Bleu: %s" % (str(bleu)))
    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    print("F_measure: %s Recall: %s Precision: %s\n" % (str(F_measure), str(recall), str(precision)))

    r2g = open('../data/relation2group.txt', 'r')
    tfidf, cc = 0.0, 0
    for line in r2g:
        items = line.strip().split()
        if len(items) > 2:
            cc += 1
            tmp = []
            for item in items:
                tmp.append(pred_all[int(item)])
            try:
                sm = SentenceSimilarity(tmp)
                sm.TfidfModel()
                tfidf += sm.similarity()
            except ValueError:
                pass
            else:
                pass
    print("number of question clusters (under the same predicate): " + str(cc))
    tfidf /= cc
    print("Tf-idf DIVERSE: %s" % str(tfidf))


if __name__ == '__main__':
    print('extracting answer patterns from training set ...')
    getAnswerPatten()
    get_rel2group()
    print('done ...')
    pattern_baseline()
print(BLEU('candidate.txt', 'reference.txt'))
