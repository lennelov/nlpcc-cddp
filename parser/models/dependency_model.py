#/usr/bin/env python
#-*- coding: utf-8 -*-


import os
import sys
import tensorflow as tf
import numpy as np

import lib.layers as layers
import lib.utils.config
import model_base

class DependencyModel(model_base.ModelBase):
    """Dependency Model"""
    def __init__(self, config):
        super(DependencyModel, self).__init__(config)

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
            'MultiClassificationMetricr': lib.layers.MultiClassificationMetricLayer(),
            'NERMetric': lib.layers.NERMetricLayer(),
        }

        # metric
        if hasattr(self.config, 'metric'):
            assert self.config.metric in metric_map, 'Unknown metric type %s' % self.config.metric
            metric_layer = metric_map[self.config.metric]
        else:
            metric_layer = lib.layers.MultiClassificationMetricLayer()

        self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes)

    def build_graph(self, inputs, mode):
        config = self.config
        # print inputs
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Embedding Layer
        if not config.use_word_pretrain_emb:
            emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='emb')
        else:
            emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size, config.emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb')

        # Build graph
        tokens = inputs['word']
        label = tf.cast(inputs['head'], tf.int32)
        self.labels = label
        nwords = tf.cast(inputs['nwords'], tf.int32)
        n_classes = self.config.n_classes

        emb = emb_layer(tokens)
        emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        # h, _ = lstm_layer(emb, x_length=nwords)
        emb = tf.transpose(emb, perm=[1, 0, 2])
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        with tf.variable_scope('lstm1'):
            output_fw, _ = lstm_cell_fw(emb, dtype=tf.float32, sequence_length=nwords)
            output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
        with tf.variable_scope('lstm2'):
            output_bw, _ = lstm_cell_bw(emb, dtype=tf.float32, sequence_length=nwords)
            output_bw = tf.layers.dropout(output_bw, rate=config.dropout_rate, training=training)
        output = tf.concat([output_fw, output_bw], axis=-1)
        # output = output_bw
        output = tf.transpose(output, perm=[1, 0, 2])

        # output = tf.layers.dropout(output, rate=config.dropout_rate, training=training)
        h = tf.concat(output, axis=2)

        # seq_logits = seq_fc_layer(h)
        seq_logits = tf.layers.dense(h, n_classes)
        # seq_logits = tf.reshape(seq_logits, [-1, self.config.max_length, self.config.n_classes])

        weights = tf.cast(tf.not_equal(tokens, 0), dtype=tf.float32)
        self.loss_op = tf.contrib.seq2seq.sequence_loss(seq_logits, label, weights, average_across_timesteps=True, \
                average_across_batch=True)
        self.infer_op = seq_logits

        self.logits = seq_logits
        self.logits_op = seq_logits
        
        metric_layer = lib.layers.MultiClassificationMetricLayer()
        self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes, weights=weights)

    def model_fn(self, inputs, mode):
        # Build graph
        self.build_graph(inputs, mode)
        # self.global_step = tf.train.get_global_step()
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
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
                mode, 
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

def main():
    pass

if '__main__' == __name__:
    main()

