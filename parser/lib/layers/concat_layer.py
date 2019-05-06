#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from .layer import Layer


class ConcatLayer(Layer):
    """ConcatLayer"""

    def __init__(self, name='concat', axis=1, **args):
        Layer.__init__(self, name, **args)
        self._axis = axis

    def _forward(self, *x):
        return tf.concat(x, self._axis)
