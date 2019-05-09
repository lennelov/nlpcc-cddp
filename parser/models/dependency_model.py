#/usr/bin/env python
#-*- coding: utf-8 -*-


import os
import sys
import tensorflow as tf
import numpy as np

import lib.layers as layers
import lib.utils.config
import model_base

class DependencyModel(model_base.ModelBase):
    """Dependency Model"""
    def __init__(self, config):
        super(DependencyModel, self).__init__(config)

    def build_graph(self, inputs, mode):
        config = self.config
        # print inputs
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Embedding Layer
        if not config.use_word_pretrain_emb:
            emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='emb')
        else:
            emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size, config.emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb')

        # Build graph
        tokens = inputs['word']
        label = tf.cast(inputs['head'], tf.int32)
        nwords = tf.cast(inputs['nwords'], tf.int32)
        n_classes = self.config.n_classes

        self.labels = label

        emb = emb_layer(tokens)
        emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        # h, _ = lstm_layer(emb, x_length=nwords)
        emb = tf.transpose(emb, perm=[1, 0, 2])
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        with tf.variable_scope('lstm1'):
            output_fw, _ = lstm_cell_fw(emb, dtype=tf.float32, sequence_length=nwords)
            output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
        with tf.variable_scope('lstm2'):
            output_bw, _ = lstm_cell_bw(emb, dtype=tf.float32, sequence_length=nwords)
            output_bw = tf.layers.dropout(output_bw, rate=config.dropout_rate, training=training)
        output = tf.concat([output_fw, output_bw], axis=-1)
        # output = output_bw
        output = tf.transpose(output, perm=[1, 0, 2])

        # output = tf.layers.dropout(output, rate=config.dropout_rate, training=training)
        h = tf.concat(output, axis=2)

        # seq_logits = seq_fc_layer(h)
        seq_logits = tf.layers.dense(h, n_classes)
        # seq_logits = tf.reshape(seq_logits, [-1, self.config.max_length, self.config.n_classes])

        weights = tf.cast(tf.not_equal(tokens, 0), dtype=tf.float32)
        self.loss_op = tf.contrib.seq2seq.sequence_loss(seq_logits, label, weights, average_across_timesteps=True, \
                average_across_batch=True)
        self.infer_op = seq_logits

        self.logits = seq_logits
        self.logits_op = seq_logits
        
        metric_layer = lib.layers.UASMetricLayer()
        self.metric = metric_layer(label, label, weights=weights)
