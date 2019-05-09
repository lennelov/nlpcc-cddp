#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from .layer import Layer


class SeqCrossLayer(Layer):
    """SeqCrossLayer"""

    def __init__(self, name='seq_cross', **args):
        Layer.__init__(self, name, **args)

    def _forward(self, x, y):
        # b: batch
        # h: embedding size
        # n: x length
        # m: y length
        return tf.einsum('bnh,bmh->bnm', x, y)


class SeqCosineLayer(Layer):
    def __init__(self, name='seq_cross', **args):
        Layer.__init__(self, name, **args)

    def _forward(self, x, y):
        # b: batch
        # h: embedding size
        # n: x length
        # m: y length
        x_norm = tf.nn.l2_normalize(x, 2)
        y_norm = tf.nn.l2_normalize(y, 2)
        return tf.einsum('bnh,bmh->bnm', x_norm, y_norm)


class SeqMatchLayer(Layer):
    def __init__(self, channel=3, emb_size=256, name='seq_match', with_bias=False, **args):
        Layer.__init__(self, name, **args)
        self.channel = channel
        self.M = self.get_variable(self._name + "_M", shape=[self.channel, emb_size, emb_size],
                                   dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.with_bias = with_bias
        if with_bias:
            self.b = self.get_variable('b', shape=[self.channel])

    def _forward(self, x, y, order='bcxy'):
        assert order in ['bcxy', 'bxyc']
        if order == 'bcxy':
            #out = tf.einsum('abd,fde,ace->afbc', x, self.M, y)
            out = tf.einsum('bxi,cij,byj->bcxy', x, self.M, y)
        else:
            out = tf.einsum('bxi,cij,byj->bxyc', x, self.M, y)

        if self.with_bias:
            out = out + self.b
        return out

