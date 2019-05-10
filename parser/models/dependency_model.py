#/usr/bin/env python
#-*- coding: utf-8 -*-


import os
import sys
import tensorflow as tf
import numpy as np

import utils.layers as layers
import utils.tools.config
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

        pos_emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='pos_emb')

        # Build graph
        batch_sze = inputs['word'].get_shape().as_list()[1]
        tokens = tf.concat([tf.constant(3,shape=(batch_sze,1)),inputs['word']], axis=-1)
        pos = tf.concat([tf.constant(3,shape=(batch_sze,1)),inputs['upos']], axis=-1)
        nwords = tf.cast(inputs['nwords'], tf.int32)

        if mode != tf.estimator.ModeKeys.PREDICT:
            label = tf.cast(inputs['head'], tf.int32)
            label_ex = tf.concat([tf.constant(0, shape=(batch_sze, 1)), label], axis=-1)
        # n_classes = self.config.n_classes

        emb = emb_layer(tokens)
        pos_emb = pos_emb_layer(pos)
        emb = tf.concat([emb, pos_emb], axis=-1)
        emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        # h, _ = lstm_layer(emb, x_length=nwords)
        emb = tf.transpose(emb, perm=[1, 0, 2])
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        with tf.variable_scope('lstm1'):
            output_fw, _ = lstm_cell_fw(emb, dtype=tf.float32, sequence_length=nwords+1)
            output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
        with tf.variable_scope('lstm2'):
            output_bw, _ = lstm_cell_bw(emb, dtype=tf.float32, sequence_length=nwords+1)
            output_bw = tf.layers.dropout(output_bw, rate=config.dropout_rate, training=training)
        output = tf.concat([output_fw, output_bw], axis=-1)
        # output = output_bw
        output = tf.transpose(output, perm=[1, 0, 2])
        # output = tf.layers.dropout(output, rate=config.dropout_rate, training=training)
        h = tf.concat(output, axis=2)
        weights = tf.cast(tf.not_equal(tokens[:,1:], 0), dtype=tf.float32)

        score = tf.map_fn(fn=compute_matrix, elems=(h, nwords), dtype=tf.float32)
        metric_layer = utils.layers.UASMetricLayer()

        if mode != tf.estimator.ModeKeys.PREDICT:
            pred = parse_proj(score, gold=label)
            pred_ex = tf.concat([tf.constant(0, shape=(batch_sze, 1)), pred], axis=-1)

            tmp_gold = tf.concat([h, tf.expand_dims(tf.cast(label_ex, dtype=tf.float32),-1)], axis=-1)
            proj_h_gold = tf.map_fn(fn=lambda inp: tf.concat([inp[:,:-1],
                    tf.map_fn(fn=lambda ip: inp[tf.cast(ip[-1], tf.int32),:-1],
                            elems=inp,
                            dtype=tf.float32)], axis=-1),
                elems=tmp_gold,
                dtype=tf.float32)
            proj_h_gold = tf.reshape(proj_h_gold, [-1, self.config.max_length + 1, 2*self.config.hidden_size])
            hidden_state_gold = tf.layers.dense(proj_h_gold[:,1:,:], config.fc_hidden_size, activation='tanh')
            score_gold = tf.layers.dense(hidden_state_gold, 1, use_bias=False)
            score_gold = tf.reduce_sum(tf.multiply(score_gold, weights))

            tmp_pred = tf.concat([h, tf.expand_dims(tf.cast(pred_ex, dtype=tf.float32),-1)], axis=-1)
            proj_h_pred = tf.map_fn(fn=lambda inp: tf.concat([inp[:,:-1],
                    tf.map_fn(fn=lambda ip: inp[tf.cast(ip[-1], tf.int32),:-1],
                            elems=inp,
                            dtype=tf.float32)], axis=-1),
                elems=tmp_pred,
                dtype=tf.float32)
            proj_h_pred = tf.reshape(proj_h_pred, [-1, self.config.max_length + 1, 2*self.config.hidden_size])
            hidden_state_pred = tf.layers.dense(proj_h_pred[:,1:,:], config.fc_hidden_size, activation='tanh')
            
            score_pred = tf.layers.dense(hidden_state_pred, 1, use_bias=False)
            score_pred = tf.reduce_sum(tf.multiply(score_pred, weights))

            self.loss_op = tf.reduce_max([tf.constant(0, dtype=tf.float32), 1- score_pred + score_gold])

        pred = parse_proj(score)
        self.metric = metric_layer(pred, label, weights=weights)
        self.infer_op = pred

        # seq_logits = tf.layers.dense(h[:,1:,:], 200)
        # # seq_logits = tf.reshape(seq_logits, [-1, self.config.max_length, self.config.n_classes])
        # # self.loss_op = tf.contrib.seq2seq.sequence_loss(seq_logits, label, weights, average_across_timesteps=True, \
        #         # average_across_batch=True)

        # self.logits = seq_logits
        # self.logits_op = seq_logits


def compute_matrix(h, sess):
    length = h.get_shape().as_list()[1]
    matrix = np.zeros([length, length])
    for i in range(length):
        for j in range(length):
            matrix[i][j] = mlp_score(h[:,i], h[:,j]).eval(session=sess)
    matrix = tf.convert_to_tensor(matrix)
return matrix



import numpy as np
import sys
from collections import defaultdict, namedtuple
from operator import itemgetter


def parse_proj(scores, gold=None):
    '''
    Parse using Eisner's algorithm.
    '''
    nr, nc = np.shape(scores)
    if nr != nc:
        raise ValueError("scores must be a squared matrix with nw+1 rows")

    N = nr - 1 # Number of words (excluding root).

    # Initialize CKY table.
    complete = np.zeros([N+1, N+1, 2]) # s, t, direction (right=1). 
    incomplete = np.zeros([N+1, N+1, 2]) # s, t, direction (right=1). 
    complete_backtrack = -np.ones([N+1, N+1, 2], dtype=int) # s, t, direction (right=1). 
    incomplete_backtrack = -np.ones([N+1, N+1, 2], dtype=int) # s, t, direction (right=1).

    incomplete[0, :, 0] -= np.inf

    # Loop from smaller items to larger items.
    for k in xrange(1,N+1):
        for s in xrange(N-k+1):
            t = s+k
            
            # First, create incomplete items.
            # left tree
            incomplete_vals0 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[t, s] + (0.0 if gold is not None and gold[s]==t else 1.0)
            incomplete[s, t, 0] = np.max(incomplete_vals0)
            incomplete_backtrack[s, t, 0] = s + np.argmax(incomplete_vals0)
            # right tree
            incomplete_vals1 = complete[s, s:t, 1] + complete[(s+1):(t+1), t, 0] + scores[s, t] + (0.0 if gold is not None and gold[t]==s else 1.0)
            incomplete[s, t, 1] = np.max(incomplete_vals1)
            incomplete_backtrack[s, t, 1] = s + np.argmax(incomplete_vals1)

            # Second, create complete items.
            # left tree
            complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
            complete[s, t, 0] = np.max(complete_vals0)
            complete_backtrack[s, t, 0] = s + np.argmax(complete_vals0)
            # right tree
            complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
            complete[s, t, 1] = np.max(complete_vals1)
            complete_backtrack[s, t, 1] = s + 1 + np.argmax(complete_vals1)
        
    value = complete[0][N][1]
    heads = [-1 for _ in range(N+1)] #-np.ones(N+1, dtype=int)
    backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads)

    value_proj = 0.0
    for m in xrange(1,N+1):
        h = heads[m]
        value_proj += scores[h,m]

    return heads


def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads):
    '''
    Backtracking step in Eisner's algorithm.
    - incomplete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *incomplete* spans.
    - complete_backtrack is a (NW+1)-by-(NW+1) numpy array indexed by a start position,
    an end position, and a direction flag (0 means left, 1 means right). This array contains
    the arg-maxes of each step in the Eisner algorithm when building *complete* spans.
    - s is the current start of the span
    - t is the current end of the span
    - direction is 0 (left attachment) or 1 (right attachment)
    - complete is 1 if the current span is complete, and 0 otherwise
    - heads is a (NW+1)-sized numpy array of integers which is a placeholder for storing the 
    head of each word.
    '''
    if s == t:
        return
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads)
            return
        else:
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads)
            return
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            heads[s] = t
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
        else:
            heads[t] = s
            backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads)
            backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads)
            return
