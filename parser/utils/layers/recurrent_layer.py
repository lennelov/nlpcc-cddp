#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
import collections

from .layer import Layer


class RecurrentLayer(Layer):
    def __init__(self, hidden_size):
        pass
        # TODO


class BiLSTMLayer(Layer):
    """BiLSTMLayer"""

    def __init__(self, hidden_size, name="BiLSTM", initializer=None, **kwargs):
        Layer.__init__(self, name, **kwargs)
        self.fw_cell = tf.contrib.rnn.LSTMCell(hidden_size, initializer=initializer, name=name + 'fw/')
        self.bw_cell = tf.contrib.rnn.LSTMCell(hidden_size, initializer=initializer, name=name + 'bw/')

    # x: batch_size * max_length * embedding_size, x_length: batch_size
    def _forward(self, x, x_length=None):
        output, state = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, x, x_length, dtype=tf.float32)
        # batch_size * max_length * 2 hidden_size
        output = tf.concat(output, axis=2)
        # batch_size * 2 hidden_size
        state = tf.concat([x[0] for x in state], axis=1)
        return output, state


class BiGRULayer(Layer):
    """BiGRULayer"""

    def __init__(self, hidden_size, initializer=None, name="BiGRULayer", **kwargs):
        Layer.__init__(self, name, **kwargs)
        self.fw_cell = tf.contrib.rnn.GRUCell(hidden_size,
                                              kernel_initializer=initializer,
                                              bias_initializer=initializer,
                                              name=name + 'fw/')
        self.bw_cell = tf.contrib.rnn.GRUCell(hidden_size,
                                              kernel_initializer=initializer,
                                              bias_initializer=initializer,
                                              name=name + 'bw/')

    # x: batch_size * max_length * embedding_size, x_length: batch_size
    def _forward(self, x, x_length=None):
        output, state = tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, x, x_length, dtype=tf.float32)
        # print(output, state)
        # batch_size * max_length * 2 hidden_size
        output = tf.concat(output, axis=2)
        # batch_size * 2 hidden_size
        state = tf.concat(state, axis=1)
        return output, state

class CompositeGRULayer(Layer):
    """CompositeGRULayer"""

    def __init__(self, component_size, hidden_size, batch_size,
                 name="CompositeGRULayer", **kwargs):
        Layer.__init__(self, name, **kwargs)
        c_shape = [component_size, hidden_size, hidden_size]
        shape = [hidden_size, hidden_size]
        self.Uz = tf.get_variable(name + "_Uz", shape=shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.Wz = tf.get_variable(name + "_Wz", shape=c_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.Ur = tf.get_variable(name + "_Ur", shape=shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.Wr = tf.get_variable(name + "_Wr", shape=c_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.Uh = tf.get_variable(name + "_Uh", shape=shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.Wh = tf.get_variable(name + "_Wh", shape=c_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.init_states = tf.constant(
            0.0,
            shape=[
                batch_size,
                hidden_size],
            dtype=tf.float32)

    def _forward(self, seq_c, seq_e):
        # To time-major
        seq_c = tf.transpose(seq_c, perm=[1, 0, 2])
        seq_e = tf.transpose(seq_e, perm=[1, 0, 2])
        seq_length = seq_c.get_shape()[0]

        def while_cond(time, state, output_ta):
            return tf.less(time, seq_length)

        def forward_step(time, s_old, output_ta):
            Uz = self.Uz
            Wz = tf.einsum('bc,cij -> bij', seq_c[time], self.Wz)
            Ur = self.Ur
            Wr = tf.einsum('bc,cij -> bij', seq_c[time], self.Wr)
            Uh = self.Uh
            Wh = tf.einsum('bc,cij -> bij', seq_c[time], self.Wh)

            z = tf.nn.sigmoid(
                tf.matmul(
                    seq_e[time],
                    Uz) +
                tf.einsum(
                    'bi,bij -> bj',
                    s_old,
                    Wz))
            r = tf.nn.sigmoid(
                tf.matmul(
                    seq_e[time],
                    Ur) +
                tf.einsum(
                    'bi,bij -> bj',
                    s_old,
                    Wr))
            h = tf.nn.sigmoid(
                tf.matmul(
                    seq_e[time],
                    Uh) +
                tf.einsum(
                    'bi,bij -> bj',
                    s_old *
                    r,
                    Wh))
            s_new = (1 - z) * h + z * s_old
            output_ta = output_ta.write(time, h)
            return (time + 1, s_new, output_ta)

        output_ta = tf.TensorArray(
            tf.float32, size=1, dynamic_size=True, infer_shape=False)
        time = tf.constant(0)
        loop_vars = [time, self.init_states, output_ta]

        _, state, output_ta = tf.while_loop(while_cond,
                                            forward_step,
                                            loop_vars)

        # To batch-major
        out = tf.transpose(output_ta.stack(), perm=[1, 0, 2])
        return out


class SpatialGRULayer(Layer):
    """SpatialGRULayer"""

    def __init__(self, channel=3, units=50, activation=tf.tanh, recurrent_activation=tf.sigmoid,
                 name="SpatialGRULayer", **kwargs):
        Layer.__init__(self, name, **kwargs)
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.channel = channel
        self.input_dim = self.channel + 3 * self.units
        self.W = self.get_variable(self._name + "_W", shape=[self.input_dim, self.units * 7], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.U = self.get_variable(self._name + "_U", shape=[self.units * 3, self.units], dtype=tf.float32,
                                   initializer=tf.orthogonal_initializer)
        self.bias = self.get_variable(self._name + "_b", shape=[self.units * 8], dtype=tf.float32,
                                      initializer=tf.zeros_initializer)
        self.wr = self.W[:, :self.units * 3]
        self.br = self.bias[:self.units * 3]
        self.wz = self.W[:, self.units * 3: self.units * 7]
        self.bz = self.bias[self.units * 3: self.units * 7]
        self.w_ij = self.get_variable(self._name + "_Wij", shape=[self.channel, self.units], dtype=tf.float32,
                                      initializer=tf.orthogonal_initializer)
        self.b_ij = self.bias[self.units * 7:]

    def _forward(self, inputs):
        input_shape = inputs.get_shape().as_list()
        self.text1_maxlen = input_shape[2]
        self.text2_maxlen = input_shape[3]
        self.recurrent_step = self.text1_maxlen * self.text2_maxlen

        def _time_distributed_dense(w, x, b):
            x = tf.keras.backend.dot(x, w)
            x = tf.keras.backend.bias_add(x, b)
            return x

        def softmax_by_row(z):
            z_transform = tf.transpose(tf.reshape(z, [-1, 4, self.units]), perm=[0, 2, 1])
            for i in range(0, self.units):
                begin1 = [0, i, 0]
                size = [-1, 1, -1]
                if i == 0:
                    z_s = tf.nn.softmax(tf.slice(z_transform, begin1, size))
                else:
                    z_s = tf.concat([z_s, tf.nn.softmax(tf.slice(z_transform, begin1, size))], 1)
            zi, zl, zt, zd = tf.unstack(z_s, axis=2)
            return zi, zl, zt, zd

        def while_cond(inputs_ta, states, step, hij, h0):
            return tf.less(step, self.recurrent_step)

        def forward_step(inputs_ta, states, step, hij, h0):
            i = tf.div(step, tf.constant(self.text2_maxlen))
            j = tf.mod(step, tf.constant(self.text2_maxlen))

            h_diag = states.read(i * (self.text2_maxlen + 1) + j)
            h_top = states.read(i * (self.text2_maxlen + 1) + j + 1)
            h_left = states.read((i + 1) * (self.text2_maxlen + 1) + j)

            s_ij = inputs_ta.read(step)
            q = tf.concat([tf.concat([h_top, h_left], 1), tf.concat([h_diag, s_ij], 1)], 1)
            r = self.recurrent_activation(_time_distributed_dense(self.wr, q, self.br))
            z = _time_distributed_dense(self.wz, q, self.bz)
            zi, zl, zt, zd = softmax_by_row(z)
            hij_ = self.activation(_time_distributed_dense(self.w_ij, s_ij, self.b_ij) +
                                   tf.keras.backend.dot(r * (tf.concat([h_left, h_top, h_diag], 1)), self.U))
            hij = zl * h_left + zt * h_top + zd * h_diag + zi * hij_
            states = states.write(((i + 1) * (self.text2_maxlen + 1) + j + 1), hij)
            hij.set_shape(h_top.get_shape())
            return inputs_ta, states, step + 1, hij, h0

        input_x = tf.transpose(inputs, [2, 3, 0, 1])
        input_x = tf.reshape(input_x, [-1, self.channel])
        input_x = tf.split(axis=0, num_or_size_splits=self.text1_maxlen * self.text2_maxlen, value=input_x)
        inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.text1_maxlen * self.text2_maxlen, name='input_ta')
        states_ta = tf.TensorArray(dtype=tf.float32, size=(self.text1_maxlen + 1) * (self.text2_maxlen + 1),
                                   name='state_ta', clear_after_read=False)

        self.bounder_state_h0 = tf.zeros([tf.shape(inputs)[0], self.units])
        for i in range(self.text2_maxlen + 1):
            states_ta = states_ta.write(i, self.bounder_state_h0)
        for i in range(self.text1_maxlen):
            states_ta = states_ta.write((i + 1) * (self.text2_maxlen + 1), self.bounder_state_h0)
        inputs_ta = inputs_ta.unstack(input_x)
        loop_vars = [inputs_ta, states_ta, tf.Variable(0, dtype=tf.int32), self.bounder_state_h0, self.bounder_state_h0]

        _, _, _, hij, _ = tf.while_loop(
            while_cond,
            forward_step,
            loop_vars
        )
        return hij


def main():
    comp_gru_layer = CompositeGRULayer(20, 128, batch_size=256)
    seq_c_ph = tf.placeholder(tf.float32, shape=[256, 32, 20])
    seq_e_ph = tf.placeholder(tf.float32, shape=[256, 32, 128])
    out = comp_gru_layer((seq_c_ph, seq_e_ph))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run([out], feed_dict={
            seq_e_ph: np.ones(shape=[256, 32, 128]),
            seq_c_ph: np.zeros(shape=[256, 32, 20])})


if '__main__' == __name__:
    main()
