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
        config = self.config
        self.inputs = inputs
        logger.info('Checking inputs ...')

    def model_fn(self, inputs, mode):
        # Build graph
        self.build_graph(inputs, mode)
        # self.global_step = tf.train.get_global_step()
        self.global_step = tf.get_variable(name="global_step", initializer=0, trainable=False)
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
            return tf.estimator.EstimatorSpec(
                mode=mode, 
                predictions=predictions, 
                export_outputs=export_outputs
                )

        # Compute evaluation op.
        # self.metrics = metrics
        if mode == tf.estimator.ModeKeys.EVAL:
            self.eval_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=self.loss_op,
                eval_metric_ops=metrics,
                predictions=predictions,
            )
            return self.eval_spec

        # Create training op.
        assert mode == tf.estimator.ModeKeys.TRAIN

        optimizer_map = {'SGD': tf.train.GradientDescentOptimizer,
                         'Adam': tf.train.AdamOptimizer,
                         'Momentum': tf.train.MomentumOptimizer}
        assert self.config.optimizer in optimizer_map.keys()
        O = optimizer_map[self.config.optimizer]
        learning_rate = tf.train.exponential_decay(self.config.learning_rate,
                tf.train.get_global_step(), self.config.decay_step, self.config.decay_rate, staircase=True)
        if self.config.optimizer == 'Momentum':
            optimizer = O(learning_rate=learning_rate, momentum=0.95)
        else:
            optimizer = O(learning_rate=learning_rate)
        
        tvars = tf.trainable_variables()
        if hasattr(self.config, 'use_clip_by_norm') and self.config.use_clip_by_norm:
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, tvars), clip_norm=1)
            if self.config.optimizer == 'Momentum':
                train_op = optimizer.apply_gradients(
                    zip(grads, tvars),
                    global_step=tf.train.get_global_step(),
                    )
            else:
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
                                                     loss=self.loss_op, 
                                                     train_op=train_op, 
                                                     eval_metric_ops=self.metric,
                                                     predictions=predictions, 
                                                     training_hooks=[logging_hook])

        return self.train_spec
