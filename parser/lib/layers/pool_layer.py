#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from .layer import Layer


class DynamicPoolLayer(Layer):
    def __init__(self):
        pass

    def _forward(self, seq_emb, actual_len):
        pass

    @staticmethod
    def _get_rescaled_index(actual_len, max_len):
        def get_index(actual_len, max_len):
            # Rescale trick, e.g.:
            # actual_len: 4, max_len: 6
            # 0, 1, 2, 3, 4, 5 -> 0, 0, 1, 2, 3, 3
            stride = 1.0 * max_len / actual_len
            index = [int(i / stride) for i in xrange(max_len)]
            return index

        result = []
        batch_size = len(actual_len)
        for idx in xrange(batch_size):
            batch_idx = np.ones([max_len]) * idx
            index = get_index(actual_len[idx], max_len)
            result.append(np.stack([batch_idx, index]))
        return result


class DynamicPool2DLayer(Layer):
    def __init__(self, kernel, stride, batch_size_placeholder,
                 name='dynamic_pool', max_row_len=33, max_col_len=32, **args):
        Layer.__init__(self, name, **args)
        self._kernel = kernel
        self._stride = stride
        self.batch_size_placeholder = batch_size_placeholder

        mesh_idx1 = np.zeros([max_row_len, max_row_len, max_col_len, max_col_len], 'int64')
        mesh_idx2 = np.zeros([max_row_len, max_row_len, max_col_len, max_col_len], 'int64')
        for i in range(1, max_row_len, 1):
            for j in range(1, max_row_len, 1):
                stride1 = float(max_col_len) / i
                stride2 = float(max_col_len) / j
                idx1_one = np.arange(max_col_len, dtype='float64') / stride1
                idx2_one = np.arange(max_col_len, dtype='float64') / stride2
                mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
                mesh_idx1[i, j, :, :] = idx1_one
                mesh_idx2[i, j, :, :] = idx2_one
        self.mesh_idx1 = tf.constant(mesh_idx1, dtype=tf.int64)
        self.mesh_idx2 = tf.constant(mesh_idx2, dtype=tf.int64)

    def _get_rescaled_index(self, len1, len2, max_row_len, max_col_len):
        batch_idx = tf.tile(
            tf.expand_dims(
                tf.expand_dims(
                    tf.range(
                        self.batch_size_placeholder, dtype=tf.int64), axis=1), axis=2), [
                1, max_col_len, max_col_len])
        len_id = tf.stack([len1, len2], axis=1)
        mesh1 = tf.gather_nd(self.mesh_idx1, len_id)
        mesh2 = tf.gather_nd(self.mesh_idx2, len_id)
        index = tf.transpose(
            tf.stack([batch_idx, mesh1, mesh2], axis=1), (0, 3, 2, 1))
        return index

    def _forward(self, mat, actual_row_len, actual_col_len):
        shape = mat.get_shape()
        max_row_len = int(shape[1])
        max_col_len = int(shape[2])

        index = self._get_rescaled_index(actual_row_len,
                                         actual_col_len,
                                         max_row_len,
                                         max_col_len)

        rescaled_mat = tf.gather_nd(mat, index)
        pool = tf.nn.max_pool(
            rescaled_mat,
            self._kernel,
            self._stride,
            'VALID')
        return pool
