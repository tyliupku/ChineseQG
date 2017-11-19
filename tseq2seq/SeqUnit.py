#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-27 下午8:37
# @Author  : Tianyu Liu

import tensorflow as tf
from Unit import *
import pickle
from AttentionUnit import AttentionWrapper
from tensorflow.python.util import nest


class SeqUnit(object):
    def __init__(self, batch_size, hidden_size, emb_size, source_vocab, target_vocab, scope_name, name, attention, lr,
                 start_token=2, stop_token=2, max_length=40):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.grad_clip = 5.0
        self.start_token = start_token
        self.stop_token = stop_token
        self.max_length = max_length
        self.scope_name = scope_name
        self.name = name
        self.attention = attention

        self.units = {}
        self.params = {}

        self.encoder_input = tf.placeholder(tf.int32, [None, None])
        self.decoder_input = tf.placeholder(tf.int32, [None, None])
        self.encoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_len = tf.placeholder(tf.int32, [None])
        self.decoder_output = tf.placeholder(tf.int32, [None, None])

        with tf.variable_scope(scope_name):
            self.enc_lstm = LstmUnit(self.hidden_size, self.emb_size, 'encoder_lstm')
            self.dec_lstm = LstmUnit(self.hidden_size, self.emb_size, 'decoder_lstm')
            self.dec_out = OutputUnit(self.hidden_size, self.target_vocab, 'decoder_output')

        self.units.update({'encoder_lstm': self.enc_lstm,'decoder_lstm': self.dec_lstm,
                           'decoder_output': self.dec_out})


        with tf.device('/cpu:0'):
            with tf.variable_scope(scope_name):
                self.embedding = tf.get_variable('embedding', [self.source_vocab, self.emb_size])
                self.encoder_embed = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
                self.decoder_embed = tf.nn.embedding_lookup(self.embedding, self.decoder_input)

        self.params.update({'embedding': self.embedding})

        # ======================================== encoder ======================================== #
        en_outputs, en_state = self.encoder(self.encoder_embed, self.encoder_len)
        # ======================================== decoder ======================================== #
        if self.attention:
            with tf.variable_scope(scope_name):
                print("normal attention")
                self.att_layer = AttentionWrapper(self.hidden_size, self.hidden_size, en_outputs, "attention")
                self.units.update({'attention': self.att_layer})

        de_outputs, de_state, self.wweight = self.decoder_t(en_state, self.decoder_embed, self.decoder_len)
        self.oooo = de_outputs
        self.g_tokens = self.decoder_g(en_state)
        # self.b_tokens, self.b_paths, self.bp, self.bs = self.decoder_beam(en_state, beam_size)
        # self.b_tokens = self.decoder_beam(en_state, beam_size)

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=de_outputs, labels=self.decoder_output)
        mask = tf.sign(tf.to_float(self.decoder_output))
        losses = mask * losses
        self.mean_loss = tf.reduce_mean(losses)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def encoder(self, inputs, inputs_len):
        batch_size = tf.shape(self.encoder_input)[0]
        max_time = tf.shape(self.encoder_input)[1]
        hidden_size = self.hidden_size

        time = tf.constant(0, dtype=tf.int32)
        h0 = (tf.zeros([batch_size, hidden_size], dtype=tf.float32),
              tf.zeros([batch_size, hidden_size], dtype=tf.float32))
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.enc_lstm(x_t, s_t, finished)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t+1, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t+1))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, inputs_ta.read(0), h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        return outputs, state

    def decoder_t(self, initial_state, inputs, inputs_len):
        batch_size = tf.shape(self.decoder_input)[0]
        max_time = tf.shape(self.decoder_input)[1]

        time = tf.constant(0, dtype=tf.int32)
        h0 = initial_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
        inputs_ta = inputs_ta.unstack(tf.transpose(inputs, [1,0,2]))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)
        weigh_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, weigh_ta, finished):
            o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
            # print o_t.get_shape().as_list()
            if self.attention:
                o_t, w_t = self.att_layer(o_t)
            o_t = self.dec_out(o_t, finished)
            weigh_ta = weigh_ta.write(t, w_t)
            emit_ta = emit_ta.write(t, o_t)
            finished = tf.greater_equal(t, inputs_len)
            x_nt = tf.cond(tf.reduce_all(finished), lambda: tf.zeros([batch_size, self.emb_size], dtype=tf.float32),
                                     lambda: inputs_ta.read(t))
            return t+1, x_nt, s_nt, emit_ta, weigh_ta, finished

        _, _, state, emit_ta, weigh_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, _5, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, h0, emit_ta, weigh_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        weights = tf.transpose(weigh_ta.stack(), [2,0,1])
        return outputs, state, weights

    def decoder_g(self, initial_state):
        batch_size = tf.shape(self.encoder_input)[0]

        time = tf.constant(0, dtype=tf.int32)
        h0 = initial_state
        f0 = tf.zeros([batch_size], dtype=tf.bool)
        x0 = tf.nn.embedding_lookup(self.embedding, tf.fill([batch_size], self.start_token))
        emit_ta = tf.TensorArray(dtype=tf.float32, dynamic_size=True, size=0)

        def loop_fn(t, x_t, s_t, emit_ta, finished):
            o_t, s_nt = self.dec_lstm(x_t, s_t, finished)
            if self.attention:
                o_t, w = self.att_layer(o_t)
            o_t = self.dec_out(o_t, finished)
            emit_ta = emit_ta.write(t, o_t)

            next_token = tf.arg_max(o_t, 1)
            x_nt = tf.nn.embedding_lookup(self.embedding, next_token)
            finished = tf.logical_or(finished, tf.equal(next_token, self.stop_token))
            finished = tf.logical_or(finished, tf.greater_equal(t, self.max_length))
            return t+1, x_nt, s_nt, emit_ta, finished

        _, _, state, emit_ta, _ = tf.while_loop(
            cond=lambda _1, _2, _3, _4, finished: tf.logical_not(tf.reduce_all(finished)),
            body=loop_fn,
            loop_vars=(time, x0, h0, emit_ta, f0))

        outputs = tf.transpose(emit_ta.stack(), [1,0,2])
        pred_tokens = tf.arg_max(outputs, 2)
        return pred_tokens

    def __call__(self, x, sess):
        loss, _ = sess.run([self.mean_loss, self.train_op],
                           {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len'],
                            self.decoder_input: x['dec_in'],
                            self.decoder_len: x['dec_len'], self.decoder_output: x['dec_out']})
        return loss

    def generate(self, x, sess):
        predictions = sess.run(self.g_tokens,
                              {self.encoder_input: x['enc_in'], self.encoder_len: x['enc_len']})
        return predictions

    def save(self, path):
        for u in self.units:
            self.units[u].save(path+u+".pkl")
        param_values = {}
        for param in self.params:
            param_values[param] = self.params[param].eval()
        with open(path+self.name+".pkl", 'wb') as f:
            pickle.dump(param_values, f, True)

    def load(self, path):
        for u in self.units:
            self.units[u].load(path+u+".pkl")
        param_values = pickle.load(open(path+self.name+".pkl", 'rb'))
        for param in param_values:
            self.params[param].load(param_values[param])