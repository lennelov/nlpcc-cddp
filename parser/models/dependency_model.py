#/usr/bin/env python
#-*- coding: utf-8 -*-


import os
import sys
import tensorflow as tf
import numpy as np

import utils.layers as layers
import utils.tools.config
import model_base

sys.setrecursionlimit(10000)

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
        # batch_sze = inputs['word'].get_shape().as_list()[0]
        batch_sze = self.config.batch_size
        tokens = tf.concat([tf.constant(3,shape=(batch_sze,1)),inputs['word']], axis=-1)
        pos = tf.concat([tf.constant(3,shape=(batch_sze,1)),inputs['upos']], axis=-1)
        nwords = tf.cast(inputs['nwords'], tf.int32)

        if mode != tf.estimator.ModeKeys.PREDICT:
            label = tf.cast(inputs['head'], tf.int32)
            label_ex = tf.concat([tf.constant(0, shape=(batch_sze, 1)), label], axis=-1)

        emb = emb_layer(tokens)
        pos_emb = pos_emb_layer(pos)
        emb = tf.concat([emb, pos_emb], axis=-1)
        emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

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
        output = tf.transpose(output, perm=[1, 0, 2])
        h = tf.concat(output, axis=2)

        weights = tf.cast(tf.not_equal(tokens[:,1:], 0), dtype=tf.float32)
        metric_layer = utils.layers.UASMetricLayer()

        fc_layer_1 = layers.SeqFCLayer(4*self.config.hidden_size, self.config.fc_hidden_size, name='fc_1')
        fc_layer_2 = layers.SeqFCLayer(self.config.fc_hidden_size, 1, with_bias=False, name='fc_2')

        if mode != tf.estimator.ModeKeys.PREDICT:
            # pred = parse_proj(score, gold=label)
            pred = tf.map_fn(fn=lambda inp: compute_score(inp[0], inp[1], inp[2], fc_layer_1, fc_layer_2, self.config.max_length), elems=(h, nwords, label), dtype=tf.int32)
            pred_ex = tf.concat([tf.zeros([batch_sze, 1], dtype=tf.int32), pred], axis=-1)
            # pred_ex = tf.reshape(pred_ex, [batch_sze, 241])

            tmp_gold = tf.concat([h, tf.expand_dims(tf.cast(label_ex, dtype=tf.float32),-1)], axis=-1)
            proj_h_gold = tf.map_fn(fn=lambda inp: tf.concat([inp[:,:-1],
                    tf.map_fn(fn=lambda ip: inp[tf.cast(ip[-1], tf.int32),:-1],
                            elems=inp,
                            dtype=tf.float32)], axis=-1),
                elems=tmp_gold,
                dtype=tf.float32)
            proj_h_gold = tf.reshape(proj_h_gold, [-1, self.config.max_length + 1, 4*self.config.hidden_size])
            hidden_state_gold = tf.tanh(fc_layer_1(proj_h_gold[:,1:,:]))
            score_gold = fc_layer_2(hidden_state_gold)
            score_gold = tf.reshape(score_gold, [batch_sze, self.config.max_length])
            score_gold = tf.reduce_sum(tf.multiply(score_gold, weights))

            tmp_pred = tf.concat([h, tf.expand_dims(tf.cast(pred_ex, dtype=tf.float32),-1)], axis=-1)
            # tmp_pred = tf.reshape(tmp_pred, [2,241,257])
            proj_h_pred = tf.map_fn(fn=lambda inp: tf.concat([inp[:,:-1],
                    tf.map_fn(fn=lambda ip: inp[tf.cast(ip[-1], tf.int32),:-1],
                            elems=inp,
                            dtype=tf.float32)], axis=-1),
                elems=tmp_pred,
                dtype=tf.float32)
            # proj_h_pred = tf.reshape(proj_h_pred, [2,241,512])
            proj_h_pred = tf.reshape(proj_h_pred, [-1, self.config.max_length + 1, 4*self.config.hidden_size])
            hidden_state_pred = tf.tanh(fc_layer_1(proj_h_pred[:,1:,:]))
            score_pred = fc_layer_2(hidden_state_pred)
            score_pred = tf.reshape(score_pred, [batch_sze, self.config.max_length])
            score_pred = tf.reduce_sum(tf.multiply(score_pred, weights))

            self.loss_op = tf.reduce_max([tf.constant(0, dtype=tf.float32), 1- score_pred + score_gold])

        # pred = parse_proj(score)
        self.infer_op = tf.map_fn(fn=lambda inp: compute_score(inp[0], inp[1], None, fc_layer_1, fc_layer_2, self.config.max_length), elems=(h, nwords), dtype=tf.int32)
        if mode != tf.estimator.ModeKeys.PREDICT:
            self.metric = metric_layer(self.infer_op, label, weights=weights)

        # seq_logits = tf.layers.dense(h[:,1:,:], 200)
        # # seq_logits = tf.reshape(seq_logits, [-1, self.config.max_length, self.config.n_classes])
        # # self.loss_op = tf.contrib.seq2seq.sequence_loss(seq_logits, label, weights, average_across_timesteps=True, \
        #         # average_across_batch=True)

        # self.logits = seq_logits
        # self.logits_op = seq_logits


def compute_score(h, nwords, gold, fc_layer_1, fc_layer_2, max_len):

    h = h[:nwords+1, :]

    # print h.get_shape().as_list()
    # length = h.get_shape().as_list()[0]
    # print 'length', length
    # matrix = tf.zeros([0,length])
    # for i in range(length):
    #     tmp = tf.zeros([1,0])
    #     for j in range(length):
    #         inp = tf.concat([h[i,:], h[j,:]], axis=-1)
    #         score = fc_layer_2(tf.tanh(fc_layer_1(inp[:,1:,:])))
    #         tmp = tf.concat([tmp, tf.reshape(score, [1,1])], 1)
    #     matrix = tf.concat([matrix, tf.reshape(tmp, [1,length])], 0)
    # print matrix

    length = nwords + 1
    matrix = tf.zeros([length, length])
    fn = lambda v: tf.squeeze(fc_layer_2(tf.tanh(fc_layer_1(tf.concat( [h, tf.transpose( tf.tile( tf.expand_dims(v, -1), [1, length] ), perm=[1,0] )], axis=-1)))), -1)
    matrix = tf.map_fn(fn=fn, elems=h, dtype=tf.float32)
    matrix = tf.reshape(matrix, [length, length])

    heads = parse_proj(matrix, length, max_len+1, gold)
    heads_new = tf.concat([heads[1:], tf.zeros([max_len-nwords], dtype=tf.int32)], axis=-1)
    return heads_new

# import numpy as np
# from collections import defaultdict, namedtuple
# from operator import itemgetter


def set_value(matrix, x, y, val):
    # 提取出要更新的行
    row = tf.gather(matrix, x)
    # 构造这行的新数据
    new_row = tf.concat([row[:y], [val], row[y+1:]], axis=0)
    # 使用 tf.scatter_update 方法进正行替换
    temp = matrix
    matrix = tf.concat([temp[:x],tf.expand_dims(new_row,1),temp[x+1:]], axis=-1)
    # matrix.assign(tf.scatter_update(matrix, x, new_row)) 

def parse_proj(scores, length, max_len, gold=None):
    '''
    Parse using Eisner's algorithm.
    '''
    # nr, nc = np.shape(scores)
    # if nr != nc:
    #     raise ValueError("scores must be a squared matrix with nw+1 rows")

    nr = length
    nc = nr

    N = nr - 1 # Number of words (excluding root).

    # # Initialize CKY table.
    # complete = tf.zeros([N+1, N+1, 2]) # s, t, direction (right=1). 
    # incomplete = tf.zeros([N+1, N+1, 2]) # s, t, direction (right=1). 
    # complete_backtrack = -tf.ones([N+1, N+1, 2], dtype=tf.int32) # s, t, direction (right=1). 
    # incomplete_backtrack = -tf.ones([N+1, N+1, 2], dtype=tf.int32) # s, t, direction (right=1).
    # Initialize CKY table.
    complete = tf.zeros([length, length, 2]) # s, t, direction (right=1). 
    incomplete = tf.zeros([length, length, 2]) # s, t, direction (right=1). 
    complete_backtrack = -tf.ones([length, length, 2], dtype=tf.int32) # s, t, direction (right=1). 
    incomplete_backtrack = -tf.ones([length, length, 2], dtype=tf.int32) # s, t, direction (right=1).

    # incomplete[0, :, 0] -= np.inf
    temp0 = incomplete[:,:,0]
    temp1 = incomplete[:,:,1]
    temp00 = tf.constant(-np.inf, shape=[1,max_len])
    temp01 = temp0[1:length,:]
    temp = tf.concat([temp00[:,:length], temp01], 0)
    incomplete = tf.concat([tf.expand_dims(temp, 2), tf.expand_dims(temp1, 2)], 2)

    # Loop from smaller items to larger items.
    k = tf.constant(1)
    s = tf.constant(0)
    tf.while_loop(lambda s,k: cond(s,k,N), lambda s,k: body(s,k,complete,incomplete,complete_backtrack,incomplete_backtrack,scores,gold), [s,k])

    value = complete[0][N][1]
    # heads = [-1 for _ in range(N+1)] #-np.ones(N+1, dtype=int)
    heads = -tf.ones([N+1], dtype=tf.int32)

    heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, 0, N, 1, 1, heads, max_len, True)

    value_proj = tf.constant(0.0)
    m = tf.constant(1)
    tf.while_loop(lambda m: cond2(m,N), lambda m: body2(m,heads,value_proj,scores), [m])        
    return heads

def cond(s,k,N):
    return tf.cond(s+k < N+1, lambda:True, lambda:False)

def cond2(m, N):
    return tf.cond(m < N+1, lambda:True, lambda:False)

def body(s,k,complete,incomplete,complete_backtrack,incomplete_backtrack,scores,gold):
    t = s + k
    # First, create incomplete items.
    # left tree
    if gold is not None:
        incomplete_vals0 = tf.add( tf.add(complete[s, s:t, 1], complete[(s+1):(t+1), t, 0]), tf.multiply(tf.ones([k]), tf.add(scores[t, s], tf.cond( tf.equal(gold[s], t), lambda :0.0, lambda : 1.0))))
    else:
        incomplete_vals0 = tf.add( tf.add(complete[s, s:t, 1], complete[(s+1):(t+1), t, 0]), tf.multiply(tf.ones([k]), tf.add(scores[t, s], 1.0)))
    temp0 = incomplete[:,:,0]
    temp1 = incomplete[:,:,1]
    set_value(temp0, s, t, tf.reduce_max(incomplete_vals0))
    incomplete = tf.concat([tf.expand_dims(temp0, 2), tf.expand_dims(temp1, 2)], 2)
    # incomplete[s, t, 0] = tf.reduce_max(incomplete_vals0)

    temp0 = incomplete_backtrack[:,:,0]
    temp1 = incomplete_backtrack[:,:,1]
    set_value(temp0, s, t,  s + tf.cast(tf.argmax(incomplete_vals0), dtype=tf.int32))
    incomplete_backtrack = tf.concat([tf.expand_dims(temp0, 2), tf.expand_dims(temp1, 2)], 2)
    # incomplete_backtrack[s, t, 0] = s + tf.argmax(incomplete_vals0, dtype=tf.int32)
    # right tree
    if gold is not None:
        incomplete_vals1 = tf.add( tf.add(complete[s, s:t, 1], complete[(s+1):(t+1), t, 0]), tf.multiply(tf.ones([k]), tf.add(scores[s, t], tf.cond( tf.equal(gold[t], s), lambda :0.0, lambda : 1.0))))
    else:
        incomplete_vals1 = tf.add( tf.add(complete[s, s:t, 1], complete[(s+1):(t+1), t, 0]), tf.multiply(tf.ones([k]), tf.add(scores[s, t], 1.0)))
    temp0 = incomplete[:,:,0]
    temp1 = incomplete[:,:,1]
    set_value(temp1, s, t,  tf.reduce_max(incomplete_vals1))
    incomplete = tf.concat([tf.expand_dims(temp0, 2), tf.expand_dims(temp1, 2)], 2)
    # incomplete[s, t, 1] = tf.reduce_max(incomplete_vals1)

    temp0 = incomplete_backtrack[:,:,0]
    temp1 = incomplete_backtrack[:,:,1]
    set_value(temp1, s, t,  s + tf.cast(tf.argmax(incomplete_vals1), dtype=tf.int32))
    incomplete_backtrack = tf.concat([tf.expand_dims(temp0, 2), tf.expand_dims(temp1, 2)], 2)
    # incomplete_backtrack[s, t, 1] = s + tf.cast(tf.argmax(incomplete_vals1), dtype=tf.int32)

    # Second, create complete items.
    # left tree
    complete_vals0 = complete[s, s:t, 0] + incomplete[s:t, t, 0]
    temp0 = complete[:,:,0]
    temp1 = complete[:,:,1]
    set_value(temp0, s, t,  tf.reduce_max(complete_vals0))
    complete = tf.concat([tf.expand_dims(temp0, 2), tf.expand_dims(temp1, 2)], 2)
    # complete[s, t, 0] = tf.reduce_max(complete_vals0)
    temp0 = complete_backtrack[:,:,0]
    temp1 = complete_backtrack[:,:,1]
    set_value(temp0, s, t,  s + tf.cast(tf.argmax(complete_vals0), dtype=tf.int32))
    complete_backtrack = tf.concat([tf.expand_dims(temp0, 2), tf.expand_dims(temp1, 2)], 2)
    # complete_backtrack[s, t, 0] = s + tf.cast(tf.argmax(complete_vals0), dtype=tf.int32)
    # right tree
    complete_vals1 = incomplete[s, (s+1):(t+1), 1] + complete[(s+1):(t+1), t, 1]
    temp0 = complete[:,:,0]
    temp1 = complete[:,:,1]
    set_value(temp1, s, t,  tf.reduce_max(complete_vals1))
    complete = tf.concat([tf.expand_dims(temp0, 2), tf.expand_dims(temp1, 2)], 2)
    # complete[s, t, 1] = tf.reduce_max(complete_vals1)
    temp0 = complete_backtrack[:,:,0]
    temp1 = complete_backtrack[:,:,1]
    set_value(temp1, s, t,  s + 1 + tf.cast(tf.argmax(complete_vals1), dtype=tf.int32))
    complete_backtrack = tf.concat([tf.expand_dims(temp0, 2), tf.expand_dims(temp1, 2)], 2)
    # complete_backtrack[s, t, 1] = s + 1 + tf.cast(tf.argmax(complete_vals1), dtype=tf.int32)
    s += 1
    k += 1
    return [s,k]

def body2(m,heads,value_proj,scores):
    h = heads[m]
    value_proj = tf.add(value_proj, scores[h,m])
    return m

def backtrack_eisner(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads, length, new_sample = False):
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
    # if s == t:
    #     return
    if not hasattr(backtrack_eisner, 'total_num') or new_sample:
        # backtrack_eisner.total_num = 1
        backtrack_eisner.total_num = length
    backtrack_eisner.total_num -= 1
    if backtrack_eisner.total_num < 0:
        return heads
    heads = tf.cond(tf.equal(s, t), lambda :heads, lambda :backtrack_eisner2(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads, length))
    return heads

def backtrack_eisner2(incomplete_backtrack, complete_backtrack, s, t, direction, complete, heads, length):
    if complete:
        r = complete_backtrack[s][t][direction]
        if direction == 0:
            heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 0, 1, heads, length)
            heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 0, 0, heads, length)
            return heads
        else:
            heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 0, heads, length)
            heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, r, t, 1, 1, heads, length)
            return heads
    else:
        r = incomplete_backtrack[s][t][direction]
        if direction == 0:
            # heads[s] = t
            heads = tf.concat([heads[:s], tf.expand_dims(t, -1), heads[s+1:]], axis=-1)
            heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads, length)
            heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads, length)
            return heads
        else:
            # heads[t] = s
            heads = tf.concat([heads[:t], tf.expand_dims(s, -1), heads[t+1:]], axis=-1)
            heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, s, r, 1, 1, heads, length)
            heads = backtrack_eisner(incomplete_backtrack, complete_backtrack, r+1, t, 0, 1, heads, length)
            return heads
