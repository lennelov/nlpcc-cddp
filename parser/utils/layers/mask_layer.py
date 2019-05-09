#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from .layer import Layer


class SeqMaskLayer(Layer):
    """SeqMaskLayer"""

    def __init__(self, name="seq_maks", **kwargs):
        super(SeqMaskLayer, self).__init__(name, **kwargs)

    def _forward(self, x, pad=0):
        return tf.cast(tf.greater(x, pad), dtype=tf.int32)

