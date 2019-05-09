#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from .layer import Layer


class MergeScoreNormalizeLayer(Layer):
    def __init__(self, num_score, n_class=1, name="MergeScoreNormalizeLayer", **kwargs):
        super(MergeScoreNormalizeLayer, self).__init__(name, **kwargs)
        self.num_score = num_score
        self.w = self.get_variable(name="normalize_w_score", shape=[num_score, n_class], constraint=tf.keras.constraints.NonNeg())

    def _forward(self, flat_scores):
        sum_w = tf.reduce_sum(self.w, axis=0, keep_dims=True)
        self.w /= sum_w                         # [num_score, 1]
        score = tf.matmul(flat_scores, self.w)  # [batch_size, 1]
        return score