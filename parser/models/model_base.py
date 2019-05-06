#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
import lib
import lib.layers as layers

logger = logging.getLogger(__name__)


class ModelBase(object):
    """ModelBase"""

    def __init__(self, config):
        self.config = config
        self.logits_op = None
        self.infer_op = None
        self.loss_op = None
        self.metric = None

    # Build graph
    def build_graph(self, inputs, mode):
        """build_graph
        Args:
            inputs (type):
        Returns:
            type: return value
        """
        config = self.config
        self.inputs = inputs
        logger.info('Checking inputs ...')

    def build_loss_and_metrics(self):

        margin = 1.0
        if hasattr(self.config, 'margin'):
            margin = self.config.margin

        loss_map = {
            'ReducedPairwiseHingeLoss': lib.layers.ReducedPairwiseHingeLossLayer(margin),
            'CrossEntroyLoss': lib.layers.CrossEntropyLossLayer(2)
        }

        metric_map = {
            'DefaultClassificationMetric': lib.layers.DefaultClassificationMetricLayer(),
            'DefaultRankingMetric': lib.layers.DefaultRankingMetricLayer(),
        }

        # loss
        '''
        if hasattr(self.config, 'loss'):
            assert self.config.loss in loss_map, 'Unknown loss type %s' % self.config.loss
            loss_layer = loss_map[self.config.loss]
        else:
            loss_layer = lib.layers.CrossEntropyLossLayer(2)

        self.loss_op = loss_layer(self.logits_op, self.labels)
        '''

        # metric
        if hasattr(self.config, 'metric'):
            assert self.config.metric in metric_map, 'Unknown metric type %s' % self.config.metric
            metric_layer = metric_map[self.config.metric]
        else:
            metric_layer = lib.layers.MultiClassificationMetricLayer()

        self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes)

    def model_fn(self, inputs, mode):
        # Build graph
        self.build_graph(inputs, mode)

        metrics = self.metric

        # Compute predictions op.
        predicted_classes = tf.argmax(self.infer_op, -1)
        predictions = {
            'infer': self.infer_op,
            'pred': predicted_classes
        }
        predictions.update(inputs)
        export_outputs = {'predict_output': tf.estimator.export.PredictOutput(predictions)}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

        # Compute evaluation op.
        # self.metrics = metrics
        if mode == tf.estimator.ModeKeys.EVAL:
            self.eval_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=self.loss_op,
                eval_metric_ops=metrics
            )
            return self.eval_spec

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        optimizer_map = {'SGD': tf.train.GradientDescentOptimizer,
                         'Adam': tf.train.AdamOptimizer,
                         'Momentum': tf.train.MomentumOptimizer}
        assert self.config.optimizer in optimizer_map.keys()
        O = optimizer_map[self.config.optimizer]
        if self.config.optimizer == 'Momentum':
            optimizer = O(learning_rate=self.config.learning_rate, momentum=0.95)
        else:
            optimizer = O(learning_rate=self.config.learning_rate)

        tvars = tf.trainable_variables()
        if hasattr(self.config, 'use_clip_by_norm') and self.config.use_clip_by_norm:
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, tvars), clip_norm=1)
            train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_global_step(),
            )
        else:
            train_op = optimizer.apply_gradients(
                optimizer.compute_gradients(self.loss_op, tvars),
                global_step=tf.train.get_global_step())
        self.train_op = train_op

        # Logging metrics
        log_out = {"loss": self.loss_op}
        for k in self.metric:
            log_out[k] = self.metric[k][1]

        logging_hook = tf.train.LoggingTensorHook(log_out, every_n_iter=100)
        self.train_spec = tf.estimator.EstimatorSpec(mode,
                                                     loss=self.loss_op, train_op=train_op, eval_metric_ops=self.metric,
                                                     predictions=predictions, training_hooks=[logging_hook])

        # layers.Layer.visualize()

        return self.train_spec

    def make_serving_inputs(self):
        """make_serving_inputs"""
        inputs_dict = {}
        inputs = self.inputs
        if 'CHAR' in self.mode:
            x_char = inputs['x_char']
            y_char = inputs['y_char']
            inputs_dict['x_char'] = tf.placeholder(name='x_char', shape=x_char.get_shape(), dtype=x_char.dtype)
            inputs_dict['y_char'] = tf.placeholder(name='y_char', shape=y_char.get_shape(), dtype=y_char.dtype)

        if 'WORD' in self.mode:
            x_word = inputs['x_word']
            y_word = inputs['y_word']
            inputs_dict['x_word'] = tf.placeholder(name='x_word', shape=x_word.get_shape(), dtype=x_word.dtype)
            inputs_dict['y_word'] = tf.placeholder(name='y_word', shape=y_word.get_shape(), dtype=y_word.dtype)

        label = inputs['label']
        inputs_dict['label'] = tf.placeholder(name='label', shape=label.get_shape(), dtype=label.dtype)
        return inputs_dict

    def generate_fc_layers(self, conf):
        """generate_fc_layers
        Args:
            conf (type): fc_layers conf
        Returns:
            type: return value
        """

        for i, c in enumerate(conf):
            activation = None
            if hasattr(c, 'activation'):
                activations = {'relu': tf.nn.relu, 'tanh': tf.nn.tanh}
                activation = activations[c.activation]
            l = layers.FCLayer(c.in_size, c.out_size, name='fc_%s' % i, activation=activation)
            yield l
