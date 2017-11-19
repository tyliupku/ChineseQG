#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:44
# @Author  : Tianyu Liu

import sys
import os
import tensorflow as tf
import time
import numpy as np
from Unit import *
from DataLoader import DataLoader
from PythonROUGE import PythonROUGE
from SimilarityMeasure import SentenceSimilarity
from bleu import BLEU


# model hyperparameters
tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")
tf.app.flags.DEFINE_boolean("attention", True,'attention layer or not')
tf.app.flags.DEFINE_integer("source_vocab", 3652,'vocabulary size')
tf.app.flags.DEFINE_integer("target_vocab", 3652,'vocabulary size')
tf.app.flags.DEFINE_float("lr", 0.0005, 'learning rate')  # Adam optimizer

# runtime figuration
tf.app.flags.DEFINE_string("gpu", '0', "GPU id.")
tf.app.flags.DEFINE_string("mode",'train','train or test')
tf.app.flags.DEFINE_string("dir", 'data', 'data')
tf.app.flags.DEFINE_string("save",'0','save directory')
tf.app.flags.DEFINE_string("load",'0','load directory')
tf.app.flags.DEFINE_integer("limits",0,'max data set size')
tf.app.flags.DEFINE_integer("report", 1000,'report')
tf.app.flags.DEFINE_float("lastbest", 0.0, 'last best bleu')

# vanilla seq2seq or tseq2seq
tf.app.flags.DEFINE_boolean("tseq2seq", False, 'tseq2seq or not')

FLAGS = tf.app.flags.FLAGS
last_best = 0.0

# for BLEU
gold_path = 'reference.txt'
pred_path = 'candidate.txt'

# for ROUGE
gold_for_ROUGE = '../run/evaluation/gold/question_'
if FLAGS.tseq2seq:
    pred_for_ROUGE = '../run/evaluation/pred_ts2s/question_'
else:
    pred_for_ROUGE = '../run/evaluation/pred_s2s/question_'


if FLAGS.load != "0":
    save_dir = FLAGS.dir + '/res/' + FLAGS.load + '/'
    save_file_dir = save_dir + 'files/'
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
else:
    if FLAGS.save == "0":
        prefix = str(int(time.time() * 1000))
        save_dir = '../run/res/' + prefix + '/'
        save_file_dir = save_dir + 'files/'
        os.mkdir(save_dir)
        if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)
    else:
        save_dir = FLAGS.save
        save_file_dir = save_dir + 'files/'
        if not os.path.exists(save_file_dir):
            os.mkdir(save_file_dir)

log_file = save_dir + 'log.txt'


def train(sess, dataloader, model):
    write_log("#######################################################")
    for flag in FLAGS.__flags:
        write_log(flag + " = " + str(FLAGS.__flags[flag]))
    write_log("#######################################################")
    trainset = dataloader.train_set
    k = 0
    for _ in range(FLAGS.epoch):
        loss, start_time = 0.0, time.time()
        for x in dataloader.batch_iter(trainset, FLAGS.batch_size, True):
            loss += model(x, sess)
            k += 1
            sys.stdout.write('training %.2f ...\r' % (k % FLAGS.report * 100.0 / FLAGS.report))
            sys.stdout.flush()
            if k % FLAGS.report == 0:
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f " % (k // FLAGS.report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                if FLAGS.tseq2seq:
                    write_log(evaluate_ts2s(sess, dataloader, model))
                else:
                    write_log(evaluate_s2s(sess, dataloader, model))
                model.save(save_dir)


# test result in the test mode
def test(sess, dataloader, model):
    if FLAGS.tseq2seq:
        evaluate_ts2s(sess, dataloader, model)
    else:
        evaluate_s2s(sess, dataloader, model)


# evaluate function for tseq2seq
def evaluate_ts2s(sess, dataloader, model):
    global last_best
    testset = dataloader.test_set
    
    topics = open('../data/test/topic.id', 'r').read().strip().split('\n')
    topics = [list(map(int, topic.strip().split(' '))) for topic in topics]
    pred_list = []

    k = 0
    with open(pred_path, 'w') as sw1:
        for x in dataloader.batch_iter(testset, FLAGS.batch_size, False):
            predictions = model.generate(x, sess)
            for summary in np.array(predictions):
                summary = list(summary)
                if 2 in summary:  # 2(START/END) marks the end of generation
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                ns = []
                for char in summary:
                    if char == 4:   # if char is (SUB), replace it with topic tokens
                        for top in topics[k]:
                            ns.append(top)
                    else:
                        ns.append(char)
                    ns.append(char)
                sw1.write(" ".join([str(t) for t in ns]) + '\n')
                with open(pred_for_ROUGE + str(k), 'w+') as sw2:
                    sw2.write(" ".join([str(t) for t in ns]) + '\n')
                k += 1
                pred_list.append([str(t) for t in ns])

    print("Total questions in Test:" + str(k))
    # BLEU test
    bleu = BLEU(pred_path, gold_path)
    print("Bleu: %s" % (str(bleu)))

    # ROUGE test
    pred_set = [pred_for_ROUGE + str(i) for i in range(k)]
    gold_set = [[gold_for_ROUGE + str(i)] for i in range(k)]
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
                tmp.append(pred_list[int(item)])
            try:
                sm = SentenceSimilarity(tmp)
                sm.TfidfModel()
                tfidf += sm.similarity()
            except ValueError:
                pass
            else:
                pass
    tfidf /= cc
    print("number of question clusters (under the same predicate): " + str(cc))
    print("Tf-idf DIVERSE: %s" % str(tfidf))

    result = "BLEU: %s BLEU_beam: %s \nF_measure: %s Recall: %s Precision: %s\n Tf-idf: %s\n " % \
             (str(bleu), str(0), str(F_measure), str(recall), str(precision), str(tfidf))

    if float(bleu) > last_best:
        last_best = float(bleu)
        to_word(pred_list, save_dir)
    return result


# evaluate function for vanilla seq2seq
def evaluate_s2s(sess, dataloader, model):
    global last_best
    testset = dataloader.test_set

    pred_list = []

    k = 0
    with open(pred_path, 'w') as sw1:
        for x in dataloader.batch_iter(testset, FLAGS.batch_size, False):
            predictions = model.generate(x, sess)
            for summary in np.array(predictions):
                summary = list(summary)
                if 2 in summary:  # 2(START/END) marks the end of generation
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                sw1.write(" ".join([str(t) for t in summary]) + '\n')
                with open(pred_for_ROUGE + str(k), 'w+') as sw2:
                    sw2.write(" ".join([str(t) for t in summary]) + '\n')
                k += 1
                pred_list.append([str(t) for t in summary])

    print("Total questions in Test:" + str(k))

    # BLEU test
    bleu = BLEU(pred_path, gold_path)
    print("Bleu: %s" % (str(bleu)))

    # ROUGE test
    pred_set = [pred_for_ROUGE + str(i) for i in range(k)]
    gold_set = [[gold_for_ROUGE + str(i)] for i in range(k)]
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
                tmp.append(pred_list[int(item)])
            try:
                sm = SentenceSimilarity(tmp)
                sm.TfidfModel()
                tfidf += sm.similarity()
            except ValueError:
                pass
            else:
                pass
    tfidf /= cc
    print("number of question clusters (under the same predicate): " + str(cc))
    print("Tf-idf DIVERSE: %s" % str(tfidf))

    result = "BLEU: %s BLEU_beam: %s \nF_measure: %s Recall: %s Precision: %s\n Tf-idf: %s\n " % \
             (str(bleu), str(0), str(F_measure), str(recall), str(precision), str(tfidf))

    if float(bleu) > last_best:
        last_best = float(bleu)
        to_word(pred_list, save_dir)
    return result


def write_log(s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s+'\n')


def main():
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        copy_file(save_file_dir)
        dataloader = DataLoader(FLAGS.limits, is_tseq2seq=FLAGS.tseq2seq)
        model = SeqUnit(batch_size=FLAGS.batch_size, hidden_size=FLAGS.hidden_size, emb_size=FLAGS.emb_size,
                        source_vocab=FLAGS.source_vocab, target_vocab=FLAGS.target_vocab, lr=FLAGS.lr,
                        scope_name="seq2seq", name="seq2seq", attention=FLAGS.attention)
        sess.run(tf.global_variables_initializer())
        if FLAGS.load != '0':
            model.load(save_dir)
        if FLAGS.mode == 'train':
            train(sess, dataloader, model)
        else:
            test(sess, dataloader, model)

if __name__=='__main__':
    with tf.device('/gpu:' + FLAGS.gpu):
        main()

