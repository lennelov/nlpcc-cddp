#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from .layer import Layer
from .fc_layer import FCLayer
from .fc_layer import SeqFCLayer


class SelfAttentionLayer(Layer):
    """SelfAttentionLayer"""

    def __init__(self, channels, hidden_size, emb_size,
                 name="SelfAttention", **kwargs):
        super(SelfAttentionLayer, self).__init__(name, **kwargs)
        # Ws1
        self.tanh = \
            FCLayer(
                hidden_size,
                emb_size,
                activation=tf.tanh,
                with_bias=False,
                name="Ws1")

        # Ws2
        self.softmax = FCLayer(
            channels, hidden_size, activation=tf.nn.softmax, with_bias=False, name="Ws2")

        self._Ws1 = self.get_variable(
            name + '_Ws1', shape=[emb_size, hidden_size])
        self._Ws2 = self.get_variable(
            name + '_Ws2', shape=[hidden_size, channels])
        
        self.channels = channels

    def _forward(self, seq):
        A = tf.tanh(tf.einsum('bse,eh->bsh', seq, self._Ws1))
        A = tf.einsum('bsh,hr->bsr', A, self._Ws2)
        A = tf.nn.softmax(A, 1)

        M = tf.einsum('bsr,bsh->brh', A, seq)
        M_shape = M.get_shape()         
        m = tf.reshape(M, [-1, M_shape[1] * M_shape[2]])

        p = tf.matmul(tf.transpose(A, [0, 2, 1]), A)
        p = p - tf.eye(self.channels)
        p = tf.reduce_sum(tf.square(p))
        return A, m, p


class SelfImportanceLayer(Layer):
    def __init__(self, hidden_size, with_bias=True, name='SelfImportanceLayer', **kwargs):
        super(SelfImportanceLayer, self).__init__(name, **kwargs)
        self.seq_fc = SeqFCLayer(hidden_size * 2, 1, activation=tf.tanh, with_bias=with_bias, name=name + '/seq_fc')
        self.fc = FCLayer(hidden_size, hidden_size, activation=tf.nn.relu, with_bias=with_bias, name=name + '/query_fc')

    def _forward(self, seq, mask=None):
        if mask is not None:
            mask = tf.cast(tf.expand_dims(mask, -1), dtype=tf.float32)
            seq = seq * mask
        print 'att seq', seq
        h = tf.reduce_max(seq, 1)
        h = self.fc(h)
        hs = tf.expand_dims(h, axis=1)

        tiled_hs = tf.tile(hs, [1, seq.get_shape()[1], 1])
        h_seq = tf.concat([seq, tiled_hs], axis=2)
        attention = self.seq_fc(h_seq)
        attention = tf.exp(attention)

        if mask is not None:
            attention = attention * mask 

        #attention = attention / tf.expand_dims(tf.reduce_sum(attention, 1), axis=-1)
        attention = tf.nn.softmax(attention, axis=1)
        attented_seq = attention * seq
        return attention, attented_seq

def main():
    pass


if '__main__' == __name__:
    main()
