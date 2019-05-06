#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import tensorflow as tf
from .layer import Layer


def masked_softmax(x, x_mask, axis, name='masked_softmax'):
    e = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    masked_e = e * tf.cast(x_mask, dtype=tf.float32)
    sum_masked_e = tf.maximum(tf.reduce_sum(masked_e, axis=axis, keepdims=True), 1e-7)
    return e, masked_e, sum_masked_e, tf.div(masked_e, sum_masked_e, name=name)


def get_padding(x, pad):
    return tf.cast(tf.equal(x, pad), tf.int64)


def get_position_encoding(length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
    """get position encoding define in Attention is All You Need
    Args:
        length (int):
        hidden_size (int):
        min_timescale (float):
        max_timescale (float):
    Returns:
        tensor[length, hidden_size]: Position Encoding
    """

    position = tf.to_float(tf.range(length))
    num_timescales = hidden_size // 2
    log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    return signal


class GumbelSoftmaxLayer(Layer):
    def __init__(self, name="GumbelSoftmaxLayer", **kwargs):
        super(GumbelSoftmaxLayer, self).__init__(name, **kwargs)

    def _forward(self, logits, temperature, hard=False):
        y = self._gumbel_softmax_sample(logits, temperature)
        if hard:
            y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

    def _sample_gumbel(self, shape, eps=2.22e-16):
        """ Sample from Gumbel(0, 1)
        Args:
            shape (type):
            eps (type):
        Returns:
            type: return value
        """
        U = tf.random_uniform(shape, minval=0, maxval=1)
        return -tf.log(-tf.log(U + eps) + eps)

    def _gumbel_softmax_sample(self, logits, temperature):
        """Draw a sample from the Gumbel-Softmax distribution
        Args:
            logits (type):
            temperature (type):
        Returns:
            type: return value
        """
        y = logits + self._sample_gumbel(tf.shape(logits))
        return tf.nn.softmax(y / temperature)


class CosineSimLayer(Layer):
    def __init__(self, name="CosineSimLayer", **kwargs):
        super(CosineSimLayer, self).__init__(name, **kwargs)

    def _forward(self, u, v):
        u_n = tf.nn.l2_normalize(u, 1, 1e-6)
        v_n = tf.nn.l2_normalize(v, 1, 1e-6)
        return tf.reduce_sum(u_n * v_n, 1, keepdims=True)  # [batch_size, 1]


class WeightSumLayer(Layer):
    def __init__(self, seq_len, dim,
                 initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32),
                 name="WeightSumLayer", **kwargs
                 ):
        super(WeightSumLayer, self).__init__(name, **kwargs)
        self.seq_len = seq_len
        self.dim = dim
        self.w = tf.get_variable('w', [dim, 1], initializer=initializer)

    def _forward(self, inputs):
        # dim = inputs.get_shape().as_list()[-1]
        # seq_len = inputs.get_shape().as_list()[-2]

        masks = tf.sign(tf.reduce_sum(tf.abs(inputs), axis=-1, keep_dims=True))
        padding = tf.ones_like(masks) * (- 2 ** 32 + 1)
        proj = tf.reshape(tf.matmul(tf.reshape(inputs, shape=[-1, self.dim]), self.w), shape=[-1, self.seq_len, 1])
        proj = tf.where(tf.equal(masks, 0), padding, proj)
        gate = tf.nn.softmax(proj, dim=1)  # [batch_size, seq_len, 1]
        seq_repr = tf.matmul(gate, inputs, transpose_a=True)  # [batch_size, 1, dim]
        seq_repr = tf.squeeze(seq_repr, axis=1)
        return seq_repr


class XorLayer(Layer):
    def __init__(self, name="XorLayer", **kwargs):
        super(XorLayer, self).__init__(name, **kwargs)

    def _forward(self, input_x, input_y, input_x_mask, input_y_mask, weight_list=None):
        batch_size, x_len = input_x.get_shape()
        batch_size, y_len = input_y.get_shape()
        x_exp = tf.stack([input_x] * y_len, 2)
        y_exp = tf.stack([input_y] * x_len, 1)
        x_mask_exp = tf.stack([input_x_mask] * y_len, 2)
        y_mask_exp = tf.stack([input_y_mask] * x_len, 1)

        mm_xor_bool = tf.equal(x_exp, y_exp)
        mm_xor_unmask = tf.cast(mm_xor_bool, tf.float32)

        mm_xor = mm_xor_unmask * x_mask_exp * y_mask_exp
        return tf.reshape(mm_xor, shape=[-1, x_len, y_len, 1])


class MaskMMLayer(Layer):
    """
    input_mm: (batch, x_len, y_len, c)
    input_x_mask: (batch, x_len)
    input_y_mask: (batch, y_len)
    return: ndarray (batch, x_len, y_len, c)
    """

    def __init__(self, val=0, name="mask_mm_by_val", **kwargs):
        super(MaskMMLayer, self).__init__(name, **kwargs)
        self.val = val

    def _forward(self, input_mm, input_x_mask, input_y_mask):
        batch_size, x_len = input_x_mask.get_shape()
        batch_size, y_len = input_y_mask.get_shape()
        c = input_mm.get_shape()[-1]
        x_mask_exp = tf.stack([input_x_mask] * y_len, 2)
        y_mask_exp = tf.stack([input_y_mask] * x_len, 1)
        x_mask_exp = tf.stack([x_mask_exp] * c, 3)
        y_mask_exp = tf.stack([y_mask_exp] * c, 3)

        # mm_mask = input_mm * x_mask_exp * y_mask_exp
        mm_mask = x_mask_exp * y_mask_exp
        val_exp = tf.ones_like(input_mm) * self.val
        mm_after_mask = tf.where(tf.equal(mm_mask, 0), val_exp, input_mm)
        return mm_after_mask


class RowWiseTop_K_Average_PoolingLayer(Layer):
    """
    inp_mm: [batch_size, x_len, y_len, c]
    x_mask: [batch_size, x_len]
    y_mask: [batch_size, y_len]
    """

    def __init__(self, name="RowWiseTop_K_Average_PoolingLayer", **kwargs):
        super(RowWiseTop_K_Average_PoolingLayer, self).__init__(name, **kwargs)
        min_val = -2 ** 32 + 1
        self.mm_mask_by_min_val_layer = MaskMMLayer(val=min_val, name='with min val')
        self.mm_mask_by_zero_layer = MaskMMLayer(val=0, name='with zero val')

    def _forward(self, inp_mm, x_mask, y_mask, ks=[1, 3, -1]):
        is_input_3d = False
        if len(inp_mm.get_shape()) == 3:
            is_input_3d = True
            inp_mm = tf.expand_dims(inp_mm, axis=-1)

        x_len = x_mask.get_shape()[1]
        y_len = y_mask.get_shape()[1]
        c_len = inp_mm.get_shape()[3]

        mm_mask_by_min_val = self.mm_mask_by_min_val_layer(inp_mm, x_mask, y_mask)
        mm_mask_by_zero = self.mm_mask_by_zero_layer(inp_mm, x_mask, y_mask)
        x_mask_exp = tf.cast(tf.stack([x_mask] * c_len, axis=2), dtype=tf.float32)  # [b, x, c]
        y_real_len = tf.reduce_sum(y_mask, axis=1, keep_dims=False)  # b
        y_real_len_exp = tf.stack([y_real_len] * x_len, axis=1)
        y_real_len_exp = tf.stack([y_real_len_exp] * c_len, axis=2)
        y_real_len_exp = tf.cast(y_real_len_exp, dtype=tf.float32)  # b, x_len, c

        mm_sum_by_row = tf.reduce_sum(mm_mask_by_zero, axis=2, keep_dims=False)  # b, x, 1, c
        mean_pooling_result = tf.divide(mm_sum_by_row, y_real_len_exp)

        outputs = []
        for k in ks:
            if k == -1:  # 此时就做mean pooling，由于有padding，所以没有直接用mean函数
                outputs.append(mean_pooling_result)
            else:
                assert k > 0 and k <= y_len

                top_k_input = tf.transpose(mm_mask_by_min_val, [0, 1, 3, 2])  # b, x, c, y
                top_k, top_k_index = tf.nn.top_k(top_k_input, k, sorted=True)  # b, x, c, k
                top_k = tf.transpose(top_k, [0, 1, 3, 2])  # b, x, k, c
                top_k_mean = tf.reduce_mean(top_k, axis=2, keep_dims=False)  # b, x, 1, c

                top_k_mean = tf.where(tf.less(y_real_len_exp, k), mean_pooling_result, top_k_mean)
                top_k_mean = top_k_mean * x_mask_exp
                outputs.append(top_k_mean)
        output = tf.stack(outputs, axis=2)  # b, x, len(k), c
        if is_input_3d:
            output = tf.squeeze(output, axis=3)

        return output


def main():
    tmp_layer = CosineSimLayer()
    # tf.Print()


if '__main__' == __name__:
    main()
