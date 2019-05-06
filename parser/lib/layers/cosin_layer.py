#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from .layer import Layer


class CosinLayer(Layer):
    """Cosin Layer"""

    def __init__(self, name='cosin', **args):
        Layer.__init__(self, name, **args)

    def _forward(self, x, y):
        x = tf.nn.l2_normalize(x, 1)
        y = tf.nn.l2_normalize(y, 1)
        return tf.reduce_sum(tf.multiply(x, y), axis=1)
