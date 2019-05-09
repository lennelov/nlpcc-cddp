#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class Layer(object):
    """Layer"""
    layers = []

    @staticmethod
    def visualize():
        for l in Layer.layers:
            logger.info(l)

    def __init__(self, name, mode=tf.estimator.ModeKeys.TRAIN, \
                 decay_mult=None, \
                 activation=None, \
                 dropout=None,
                 layer_normalization=False):
        self._name = name
        self._decay_mult = decay_mult
        self._activation = activation
        # self._layer_normalization = layer_normalization
        self._dropout = dropout
        self.mode = mode

        self._layer_norm_scale = []
        self._layer_norm_bias = []

        self.in_ops = {}
        self.out_ops = {}
        Layer.layers.append(self)

    def get_variable(self, name, **kwargs):
        """get_variable
        Args:
            name (str): The variable name
            **kwargs (str): tf.get_variable's kwargs
        Returns:
            tf.Variable: return value
        """
        if self._decay_mult:
            kwargs['regularizer'] = lambda x: tf.nn.l2_loss(x) * self._decay_mult

        return tf.get_variable(self._name + '/' + name, **kwargs)

    def __call__(self, *input_ops, **kwargs):
        for input_op in input_ops:
            if hasattr(input_op, 'name'):
                self.in_ops[input_op.name] = input_op

        output_ops = self._forward(*input_ops, **kwargs)

        # Postprocess 
        if self._activation:
            output_ops = self._activation(output_ops)

        if self._dropout is not None:
            output_ops = tf.nn.dropout(output_ops, dropout_keep_prob)

        output = output_ops
        if hasattr(output_ops, '__iter__'):
            output_ops = [output_ops]

        for output_op in output_ops:
            if hasattr(output_op, 'name'):
                self.out_ops[output_op.name] = output_op
        return output

    def __repr__(self):
        output = {}

        in_ops = []
        for k, v in self.in_ops.items():
            r = {'name': k, 'shape': '%s' % v.get_shape()}
            in_ops.append(r)
        output['in'] = in_ops

        out_ops = []
        for k, v in self.out_ops.items():
            r = {'name': k, 'shape': '%s' % v.get_shape()}
            out_ops.append(r)
        output['out'] = in_ops

        output = {'%s:%s' % (self.__class__.__name__, self._name): output}
        return json.dumps(output, indent=4)

    def _forward(self, *input_ops):
        return input_ops
