#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from .layer import Layer
import operator


class FCLayer(Layer):
    """FCLayer"""

    def __init__(self,
                 in_size,
                 out_size,
                 name="fc",
                 constraint=None,
                 initializer=None,
                 with_bias=True,
                 normalize_w_by_col=False,
                 trainable=True,
                 **kwargs):
        Layer.__init__(self, name, **kwargs)

        self._normalize_w_by_col = normalize_w_by_col
        if self._normalize_w_by_col:
            with_bias = False

        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._W = self.get_variable('W', shape=[in_size, out_size],
                                    initializer=initializer, constraint=constraint, trainable=trainable)
        self._b = None
        if with_bias:
            self._b = self.get_variable('b', initializer=tf.zeros([out_size]), trainable=trainable)

    def _forward(self, x):
        if self._normalize_w_by_col:
            W_norm = tf.nn.softmax(self._W, axis=0)
            y = tf.matmul(x, W_norm)
        else:
            if self._b is not None:
                y = tf.nn.xw_plus_b(x, self._W, self._b)
            else:
                y = tf.matmul(x, self._W)
        return y


class SeqFCLayer(FCLayer):
    """SeqFCLayer"""

    def __init__(self, in_size, out_size, name="seqfc", **kwargs):
        super(SeqFCLayer, self).__init__(in_size, out_size, name, **kwargs)
        self._in_size = in_size
        self._out_size = out_size

    def _forward(self, x):
        xs = x.get_shape()
        s_in = reduce(operator.mul, xs[:1], 1)
        h = tf.reshape(x, [-1, self._in_size])
        h = super(SeqFCLayer, self)._forward(h)
        s_out = [-1] + xs.dims[1:-1] + [self._out_size]
        h = tf.reshape(h, s_out)
        return h

