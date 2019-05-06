#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = 'Chao Qiao'
__copyright__ = 'Copyright (c) 2018 bytedance.com, Inc.'
__license__ = 'MIT'
__version__ = '0.0.1'
__email__ = 'qiaochao@bytedance.com'
__status__ = 'Development'

import tensorflow as tf
from lib.layers.layer import Layer
from lib.layers.attention_layer import MultiHeadsDotProductAttentionLayer
from lib.layers.fc_layer import SeqFCLayer

import tensorflow as tf


class FeedForwardNetworks(Layer):
    def __init__(self, hidden_size,
                 name='FeedForwardNetworks',
                 initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                 **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._seq_fc1 = SeqFCLayer(hidden_size, hidden_size, with_bias=True, name=name + '/fc1',
                                   initializer=initializer)
        self._seq_fc2 = SeqFCLayer(hidden_size, hidden_size, with_bias=True, name=name + '/fc2',
                                   initializer=initializer)

    def _forward(self, h):
        fc_x = tf.nn.relu(self._seq_fc2(self._seq_fc1(h)))
        h = tf.contrib.layers.layer_norm(h + fc_x)
        return h


class TransformerEncoderLayer(Layer):
    """TransformerEncoderLayer"""

    def __init__(self, d_k, d_v, n_heads, hidden_size,
                 name='transformer_encoder',
                 initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                 **kwargs):
        Layer.__init__(self, name, **kwargs)
        with tf.variable_scope(name):
            self.attention_layer = MultiHeadsDotProductAttentionLayer(
                hidden_size=hidden_size,
                d_k=d_k,
                d_v=d_v,
                n_heads=n_heads,
                name=name + '/MultiHeadAttentionLayer',
                initializer=initializer,
            )
            self.ff_layer = FeedForwardNetworks(
                hidden_size=hidden_size,
                name=name + '/FeedForwardNetworks',
                initializer=initializer,
            )

    def _forward(self, x, x_mask):
        # Multi-head attention
        _, h = self.attention_layer(x, x, x_mask, x_mask)

        # Position-wise Feed-Forward Networks
        h = self.ff_layer(h)
        return h


class TransformerDecoderLayer(Layer):
    """TransformerDecoderLayer"""

    def __init__(self, d_k, d_v, n_heads, hidden_size,
                 name='transformer_decoder',
                 initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                 **kwargs):
        Layer.__init__(self, name, **kwargs)
        with tf.variable_scope(name):
            self.self_attention_layer = MultiHeadsDotProductAttentionLayer(
                hidden_size=hidden_size,
                d_k=d_k,
                d_v=d_v,
                n_heads=n_heads,
                name=name + '/self_attention_layer'
            )

            self.multi_head_attention_layer = MultiHeadsDotProductAttentionLayer(
                hidden_size=hidden_size,
                d_k=d_k,
                d_v=d_v,
                n_heads=n_heads,
                name=name + '/multi_head_attention_layer'
            )

            self.ff_layer = FeedForwardNetworks(
                hidden_size=hidden_size,
                name=name + '/FeedForwardNetworks',
                initializer=initializer,
            )

    def _forward(self, x, y, x_mask, y_mask):
        # Multi-head self attention (x, x)
        _, x = self.self_attention_layer(x, x, x_mask, x_mask)

        # Multi-head ytox attention (x, y)
        _, h = self.self_attention_layer(x, y, x_mask, y_mask)

        # Position-wise Feed-Forward Netsworks
        h = self.ff_layer(h)
        return h


def main():
    pass


if __name__ == '__main__':
    main()
