#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from .layer import Layer


class RegionAlignmentLayer(Layer):
    """RegionAlignmentLayer"""

    def __init__(self, region_size, name="RegionAlig", **args):
        Layer.__init__(self, name, **args)
        self._region_size = region_size

    def _forward(self, x, padding='SAME'):
        """_forward
        Args:
            x (type): input sequence
            padding (type): padding mode, 'SAME' or 'VALID'
        Returns:
            type: return value
        """
        # print "x: ", x
        # print "type(x): ", type(x)

        region_radius = self._region_size / 2
        assert padding in ['SAME', 'VALID'] 

        if padding == 'SAME':
            paddings = [[region_radius, region_radius] if d == 1 else [0, 0] for d in xrange(len(x.shape))]
            x = tf.pad(x, paddings=paddings, mode='CONSTANT')

        r = xrange(region_radius, x.shape[1] - region_radius)

        # print x, x.shape, x.get_shape() 
        aligned_seq = map(
                lambda i: x[:, i - region_radius: i + region_radius + 1], r)
        
        # print aligned_seq

        # for i, t in enumerate(aligned_seq):
        #     print i, t 
    
        aligned_seq = tf.convert_to_tensor(aligned_seq)
        aligned_seq = tf.transpose(aligned_seq, perm=[1, 0, 2])
        return aligned_seq


class WindowAlignmentLayer(Layer):
    """WindowAlignmentLayer"""

    def __init__(self, window_size, name="WindowAlignment", **args):
        Layer.__init__(self, name, **args)
        self._window_size = window_size

    def _forward(self, x):
        """[1,2,3,4,5] -> [[1,2,3], [2,3,4], [3,4,5]]"""

        # batch_size, sequence_length, ...
        seq_len = x.get_shape()[1]
        win_size = self._window_size
        y = tf.map_fn(lambda i: x[:, i: i + win_size, ...],
                      tf.range(seq_len - win_size + 1), dtype=tf.float32)
        y = tf.transpose(y, perm=[1, 0, 2, 3])
        return y
