#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from .layer import Layer
from utils.tools.tf_metrics import recall, precision, f1


class DefaultClassificationMetricLayer(Layer):
    def __init__(self, name='DefaultClassificationMetricLayer', **kwargs):
        Layer.__init__(self, name, **kwargs)

    def _forward(self, logits, label, weights=None):
        actual = tf.cast(label, tf.int64)
        predicted = tf.argmax(logits, -1)
        accuracy = tf.metrics.accuracy(labels=actual, predictions=predicted, weights=weights, name='acc_op')
        recall = tf.metrics.recall(labels=actual, predictions=predicted, weights=weights, name='recall_op')
        precision = tf.metrics.precision(labels=actual, predictions=predicted, weights=weights, name='precision_op')
        auc = tf.metrics.auc(labels=actual, predictions=predicted, weights=weights, name='precision_op')

        calc_f1 = lambda p, r : p * r / (p + r) * 2
        f1 = (calc_f1(precision[0], recall[0]), calc_f1(precision[1], recall[1]))
        tf.summary.scalar('precision', precision[1])
        tf.summary.scalar('recall', recall[1])
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('f1', f1[1])
        tf.summary.scalar('auc', auc[1])
        metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}

        return metrics

class DefaultLossMetricLayer(Layer):
    def __init__(self, name='DefaultLossMetricLayer', **kwargs):
        Layer.__init__(self, name, **kwargs)

    def _forward(self, loss):
        loss_metrics = tf.metrics.mean(loss)
        tf.summary.scalar('loss', loss_metrics[1])
        metrics = {'loss': loss_metrics}
        return metrics

class MultiClassificationMetricLayer(Layer):
    def __init__(self, name='MultiClassificationMetricLayer', **kwargs):
        Layer.__init__(self, name, **kwargs)

    def _forward(self, logits, label, n_classes, weights=None):
        actual = tf.cast(label, tf.int64)
        predicted = tf.argmax(logits, -1)

        pos = [i for i in range(n_classes)]
        accuracy = tf.metrics.accuracy(labels=actual, predictions=predicted, weights=weights, name='acc_op')
        rec = recall(labels=actual, predictions=predicted, num_classes=n_classes, pos_indices=pos, average='weighted', weights=weights)
        prec = precision(labels=actual, predictions=predicted, num_classes=n_classes, pos_indices=pos, average='weighted', weights=weights)
        f1_op = f1(labels=actual, predictions=predicted, num_classes=n_classes, pos_indices=pos, average='weighted', weights=weights)

        tf.summary.scalar('precision', prec[1])
        tf.summary.scalar('recall', rec[1])
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('f1', f1_op[1])
        metrics = {'accuracy': accuracy, 'precision': prec, 'recall': rec, 'f1': f1_op}
        return metrics

class UASMetricLayer(Layer):
    def __init__(self, name='UASMetricLayer', **kwargs):
        Layer.__init__(self, name, **kwargs)

    def _forward(self, pred, label, weights=None):
        pred = tf.cast(pred, tf.int64)
        label = tf.cast(label, tf.int64)
        uas = tf.metrics.accuracy(labels=label, predictions=pred, weights=weights, name='acc_op')
        metrics = {'uas': uas}
        return metrics
