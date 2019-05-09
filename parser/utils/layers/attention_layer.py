#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf

from utils.layers.layer import Layer
from utils.layers.fc_layer import SeqFCLayer
from utils.layers.functions_layer import masked_softmax


class AttentionLayer(Layer):
    """AttentionLayer"""

    def __init__(self, hidden_size, emb_size, name="Attention", trainable=True, **kwargs):
        super(AttentionLayer, self).__init__(name, **kwargs)

        self.W = self.get_variable(name + '_W', shape=[emb_size, hidden_size], trainable=trainable)
        self.U = self.get_variable(
            name + '_U',
            shape=[
                hidden_size,
                hidden_size], trainable=trainable)
        self.V = self.get_variable(name + '_V', shape=[hidden_size, 1], trainable=trainable)

    def _forward(self, state, seq):
        temp_W = tf.einsum('bij,jk->bik', seq, self.W)

        temp_U = tf.tile(
            tf.expand_dims(tf.matmul(state, self.U), axis=1),
            [1, int(seq.get_shape()[1]), 1])

        temp = tf.tanh(temp_W + temp_U)
        x_e = tf.einsum('bij,jk->bik', temp, self.V)[:, :, 0]
        x_a = tf.nn.softmax(x_e)
        attented = seq * tf.tile(tf.expand_dims(x_a, axis=2),
                                 [1, 1, seq.get_shape()[2]])

        return x_a, attented


class SymAttentionLayer(AttentionLayer):
    def __init__(self, hidden_size, emb_size, name='SymAttention', **kwargs):
        super(SymAttentionLayer, self).__init__(
            hidden_size, emb_size, name=name, **kwargs)

    def _forward(self, state, seq):
        temp_W = tf.einsum('bij,jk->bik', seq, self.W)

        temp_U = tf.tile(
            tf.expand_dims(tf.matmul(state, self.U), axis=1),
            [1, int(seq.get_shape()[1]), 1])

        temp = tf.tanh(temp_W + temp_U)
        x_e = tf.einsum('bij,jk->bik', temp, self.V)[:, :, 0]
        x_a = tf.nn.softmax(x_e)
        sym_x_a = tf.nn.softmax(-x_e)

        attented = seq * tf.tile(tf.expand_dims(x_a, axis=2),
                                 [1, 1, seq.get_shape()[2]])

        sym_attented = seq * \
                       tf.tile(tf.expand_dims(sym_x_a, axis=2),
                               [1, 1, seq.get_shape()[2]])

        return x_a, sym_x_a, attented, sym_attented


class MultiHeadsDotProductAttentionLayer(Layer):
    def __init__(self, hidden_size, d_k, d_v, n_heads,
                 initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                 name='MultiHeadsDotProductAttentionLayer', **kwargs):
        super(MultiHeadsDotProductAttentionLayer, self).__init__(name=name, **kwargs)
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.q_dense_layer = SeqFCLayer(hidden_size, d_k * n_heads, with_bias=False, name=name + "/q",
                                        initializer=initializer)
        self.k_dense_layer = SeqFCLayer(hidden_size, d_k * n_heads, with_bias=False, name=name + "/k",
                                        initializer=initializer)
        self.v_dense_layer = SeqFCLayer(hidden_size, d_v * n_heads, with_bias=False, name=name + "/v",
                                        initializer=initializer)
        self.output_layer = SeqFCLayer(n_heads * d_v, hidden_size, with_bias=False, name=name + "/output",
                                       initializer=initializer)

    def _forward(self, x, y, x_mask, y_mask, tmp_inputs=None, pre_name=None):
        # input should be equal to output
        output_size = x.get_shape().as_list()[-1]
        # output_size = tf.shape(x)[-1]
        assert output_size == self.hidden_size

        n_heads = self.n_heads

        # Linear projections
        Q = self.q_dense_layer(x)  # (N, x_len, h * dk)
        K = self.k_dense_layer(y)  # (N, y_len, h * dk)
        V = self.v_dense_layer(y)  # (N, y_len, h * dv)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, n_heads, axis=2), axis=0)  # (N * h, x_len, dk)
        K_ = tf.concat(tf.split(K, n_heads, axis=2), axis=0)  # (N * h, y_len, dk)
        V_ = tf.concat(tf.split(V, n_heads, axis=2), axis=0)  # (N * h, y_len, dv)

        # (N * h, x_len, dk) matmul (N * h, dk, y_len) -> (h * N, x_len, y_len)
        att = tf.matmul(Q_, K_, transpose_b=True)

        # Scale
        att = att / (self.d_k ** 0.5)
        if tmp_inputs:
            tmp_inputs[pre_name + '_origin_att'] = att

        # y Masking: (N, y_len) -> (N * h, x_len, y_len)
        # x Masking: (N, x_len) -> (N * h, x_len, y_len)
        y_mask = tf.tile(tf.expand_dims(y_mask, 1), [n_heads, tf.shape(x)[1], 1])
        x_mask = tf.tile(tf.expand_dims(x_mask, 2), [n_heads, 1, tf.shape(y)[1]])

        # Activation -> (N * h, x_len, y_len)
        att_shape = tf.shape(att)
        e, masked_e, sum_masked_e, att = masked_softmax(att, y_mask, axis=-1)
        # when building graph, shape of att will be converted from [?, x_len, y_len] to [?, ?, y_len] after masked_softmax and cause potential problem, so reshape it after masked_softmax
        att = tf.reshape(att, att_shape)
        att *= x_mask
        if tmp_inputs:
            tmp_inputs[pre_name + '_e'] = e
            tmp_inputs[pre_name + '_masked_e'] = masked_e
            tmp_inputs[pre_name + '_sum_masked_e'] = sum_masked_e
            tmp_inputs[pre_name + '_att'] = att

        # (h * N, x_len, y_len) matmul (N * h, y_len, dv) ->  (N * h, x_len, dv)
        outputs = tf.matmul(att, V_)

        # Restore shape (N , x_len, dv) -> (N, x_len, dv * h)
        outputs = tf.concat(tf.split(outputs, n_heads, axis=0), axis=2)

        # Linear projections (N, x_len, dv * h)->  (N, x_len, output_size)
        outputs = self.output_layer(outputs)

        # Residual connection & Normalize
        outputs = tf.contrib.layers.layer_norm(x + outputs)
        # outputs = x + outputs

        return att, outputs


class DotProductAttentionLayer(Layer):
    def __init__(self,name='DotProductAttentionLayer', **kwargs):
        super(DotProductAttentionLayer, self).__init__(name=name, **kwargs)

    def _forward(self, x, y, x_mask, y_mask, tmp_inputs=None, pre_name=None):

        n_heads = 1

        # Linear projections
        Q = x  # self.q_dense_layer(x)  # (N, x_len, h * dk)
        K = y  # self.k_dense_layer(y)  # (N, y_len, h * dk)
        V = y  # self.v_dense_layer(y)  # (N, y_len, h * dv)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, n_heads, axis=2), axis=0)  # (N * h, x_len, dk)
        K_ = tf.concat(tf.split(K, n_heads, axis=2), axis=0)  # (N * h, y_len, dk)
        V_ = tf.concat(tf.split(V, n_heads, axis=2), axis=0)  # (N * h, y_len, dv)

        # (N * h, x_len, dk) matmul (N * h, dk, y_len) -> (h * N, x_len, y_len)
        att = tf.matmul(Q_, K_, transpose_b=True)

        # Scale
        # att = att / (self.d_k ** 0.5)
        if tmp_inputs:
            tmp_inputs[pre_name + '_origin_att'] = att

        # y Masking: (N, y_len) -> (N * h, x_len, y_len)
        # x Masking: (N, x_len) -> (N * h, x_len, y_len)
        y_mask = tf.tile(tf.expand_dims(y_mask, 1), [n_heads, tf.shape(x)[1], 1])
        x_mask = tf.tile(tf.expand_dims(x_mask, 2), [n_heads, 1, tf.shape(y)[1]])

        # Activation -> (N * h, x_len, y_len)
        e, masked_e, sum_masked_e, att = masked_softmax(att, y_mask, axis=-1)

        att *= x_mask
        if tmp_inputs:
            tmp_inputs[pre_name + '_e'] = e
            tmp_inputs[pre_name + '_masked_e'] = masked_e
            tmp_inputs[pre_name + '_sum_masked_e'] = sum_masked_e
            tmp_inputs[pre_name + '_att'] = att

        # (h * N, x_len, y_len) matmul (N * h, y_len, dv) ->  (N * h, x_len, dv)
        outputs = tf.matmul(att, V_)

        # Restore shape (N , x_len, dv) -> (N, x_len, dv * h)
        outputs = tf.concat(tf.split(outputs, n_heads, axis=0), axis=2)

        # # Linear projections (N, x_len, dv * h)->  (N, x_len, output_size)
        # outputs = self.output_layer(outputs)

        # Residual connection & Normalize
        outputs = tf.contrib.layers.layer_norm(x + outputs)
        # outputs = x + outputs

        return att, outputs


def main():
    pass


if '__main__' == __name__:
    main()
