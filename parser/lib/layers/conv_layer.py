#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from .layer import Layer


class ConvLayer(Layer):
    def __init__(self, emb_size, win_size, hidden_size, pool=True, name='Conv', **args):
        Layer.__init__(self, name, **args)
        shape = [win_size, emb_size, 1, hidden_size]

        self.emb_size = emb_size
        self.win_size = win_size
        self.hidden_size = hidden_size
        self.pool = pool
        self.W = self.get_variable('W', 
                initializer=tf.contrib.layers.xavier_initializer_conv2d(shape), shape=shape)
        self.b = self.get_variable("b", initializer=tf.constant(0.1,shape=[hidden_size]))

    def _forward(self, emb):
        emb = tf.expand_dims(emb, -1)

        h = tf.nn.conv2d(
            emb,
            self.W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        h = tf.nn.relu(tf.nn.bias_add(h, self.b), name="relu")

        # Max-pooling over the outputs
        '''
        h = tf.nn.max_pool(
            h,
            ksize=[1, h.get_shape()[1], 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        '''
        if self.pool:
            h = tf.reduce_max(h, axis=1)
            h = tf.reshape(h, [-1, self.hidden_size])

        return h


class Conv2DLayer(Layer):
    def __init__(self, filter_height, filter_wight, n_chanels, n_filters, name='Conv2d',
                 padding='SAME', **args):
        Layer.__init__(self, name, **args)

        shape = [filter_height, filter_wight, n_chanels, n_filters]

        self.W = self.get_variable('W',
                initializer=tf.contrib.layers.xavier_initializer_conv2d(shape), shape=shape)

        self.b = self.get_variable("b", initializer=tf.constant(0.1, shape=[n_filters]))

        self.padding = padding

    def _forward(self, img):
        h = tf.nn.conv2d(
            img,
            self.W,
            strides=[1, 1, 1, 1],
            padding=self.padding,
            name="conv2d")

        h = tf.nn.relu(tf.nn.bias_add(h, self.b), name="relu")
        return h


def main():
    pass


if '__main__' == __name__:
    main()
