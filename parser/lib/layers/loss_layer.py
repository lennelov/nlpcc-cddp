#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from .layer import Layer


class ReducedPairwiseHingeLossLayer(Layer):
    def __init__(self, margin, name='ReducedPairwiseHingeLossLayer', **kwargs):
        Layer.__init__(self, name, **kwargs)
        self.margin = margin

    def _forward(self, score, label=None):
        score = tf.reshape(score, [-1, 2])
        pos_score = score[:, 0]
        neg_score = score[:, 1]
        return tf.reduce_mean(tf.maximum(0., self.margin + neg_score - pos_score))


class CrossEntropyLossLayer(Layer):
    def __init__(self, n_classes, name='CrossEntropyLossLayer', **kwargs):
        Layer.__init__(self, name, **kwargs)
        self.n_classes = n_classes

    def _forward(self, logits, label):
        one_hot_labels = tf.one_hot(label, self.n_classes)
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logits))

'''
The next class is added by zhaotong which may be used to caculate loss using weight 
'''


class WeightedCrossEntropyLossLayer(CrossEntropyLossLayer):

    def _forward(self, logits, label, data_weigh=None):
        one_hot_labels = tf.one_hot(label, self.n_classes)
        if data_weigh is not None:
            return tf.reduce_mean(
                tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logits),
                            data_weigh))
        else:
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels, logits=logits))
