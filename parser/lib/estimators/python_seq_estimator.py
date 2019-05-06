#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File	: python_estimator.py

Date	: 2018-07-12 11:50

Brief	: 
"""

__author__ = 'Chao Qiao'
__copyright__ = 'Copyright (c) 2018 bytedance.com, Inc.'
__license__ = 'MIT'
__version__ = '0.0.1'
__email__ = 'qiaochao@bytedance.com'
__status__ = 'Development'

import os
import time
import datetime
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from lib import datasets

from python_estimator import PythonEstimator
from lib.utils import entity_f1
from tensorflow.python import pywrap_tensorflow

logger = logging.getLogger(__name__)


class PythonSequenceLabellingEstimator(PythonEstimator):
    """Estimator used for computing Entity F1"""

    def __init__(self, conf, model):
        super(PythonSequenceLabellingEstimator, self).__init__(conf, model)
        self.config.best_checkpoint_dir = self.config.checkpoint_dir + '/best'

    def build_graph(self, mode, use_best=False):
        self.action_mode = mode
        self.fetch_dict = {}
        self.add_estimator_inputs(mode)
        if mode == 'EXPORT':
            self.estimator_spec = self.model.model_fn(self.model_inputs, mode=tf.estimator.ModeKeys.PREDICT)
            self.fetch_dict[mode] = self.make_fetch_dict(mode)
        else:
            self.estimator_spec = self.model.model_fn(self.model_inputs, mode=mode)
            self.set_eval_and_summary()
            if mode == tf.estimator.ModeKeys.TRAIN:
                self.global_step = tf.train.get_global_step()
                self.global_epoch = 0
                self.increase_global_step = tf.assign_add(self.global_step, 1)
                if hasattr(self.config, 'save_best_result'):
                    self.build_save_best_result_graph()
                self.fetch_dict[tf.estimator.ModeKeys.EVAL] = self.make_fetch_dict(tf.estimator.ModeKeys.EVAL)

            self.fetch_dict[mode] = self.make_fetch_dict(mode)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.max_to_keep)
        logger.info('Start session ...')
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self._restore_checkpoint(use_best=use_best)

    def train(self):
        self.build_graph(mode=tf.estimator.ModeKeys.TRAIN)
        logger.info('Start training ...')
        watch_start = start_time = time.time()
        training_finished = False
        count = 0   # Early stopping
        max_step = 0
        max_score = 0
        while True:
            epoch_start_time = time.time()

            for batch in self.datasets[self.train_name].make_iter(self.config.batch_size):
                fetch_result = self.feedforward(
                    batch=batch,
                    mode=tf.estimator.ModeKeys.TRAIN,
                    name=self.train_name,
                )
                if not self.global_step:
                    continue
                if self.global_step % self.config.log_every_n_steps == 0:
                    self.log_result(
                        name=self.train_name,
                        step=self.config.log_every_n_steps,
                        speed=(time.time() - watch_start),
                        fetch_result=fetch_result,
                    )
                    watch_start = time.time()
                if hasattr(self.config, 'save_checkpoints_steps') and \
                        self.global_step % self.config.save_checkpoints_steps == 0:
                        self._save_checkpoint()

                if hasattr(self.config, 'eval_interval_steps') and \
                        self.global_step % self.config.eval_interval_steps == 0:
                    score = self.eval(is_in_train=True)
                    # print score, count, self.config.tolerance, count >= self.config.tolerance
                    if score < max_score:
                        count += 1
                    else:
                        count = 0
                        max_score = score
                        max_step = self.global_step
                        self._save_checkpoint(use_best=True)
                    if count >= self.config.tolerance:
                        training_finished = True
                        break
                    else:
                        self._save_checkpoint()

                if hasattr(self.config, 'max_training_steps') and \
                        self.global_step > self.config.max_training_steps:
                    score = self.eval(is_in_train=True)
                    print score
                    if score > max_score:
                        self._save_checkpoint(use_best=True)
                        max_score = score
                        max_step = global_step

                    training_finished = True
                    break

                # when the eval results does not have any improvement after "auto_end_time" times, end the train
                if hasattr(self.config, 'auto_end_time') and \
                        self.no_update_times >= self.config.auto_end_time:
                    training_finished = True
                    break

            if hasattr(self.config, 'save_checkpoints_epochs'):
                self._save_checkpoint()

            if hasattr(self.config, 'eval_interval_epochs'):
                score = self.eval(is_in_train=True)
                print score, count, self.config.tolerance, count >= self.config.tolerance
                if score < max_score:
                    count += 1
                else:
                    count = 0
                    max_score = score
                    self._save_checkpoint(use_best=True)
                if count >= self.config.tolerance:
                    training_finished = True
                    break
                else:
                    self._save_checkpoint()

            self.do_something_when_epoch_over(time.time() - epoch_start_time)

            if training_finished:
                break

            logger.info('Epoch %s finished, %s elapsed.', self.global_epoch,
                        datetime.timedelta(seconds=time.time() - start_time))
            self.global_epoch += 1
        logger.info('Training finished, %s elapsed.', datetime.timedelta(seconds=time.time() - start_time))

    def update_fetch_result(self, name, fetch_results, results=None):
        if not self.config.use_entity_f1:
            return fetch_results['accuracy']

        word2id = {i : line.strip() for i, line in enumerate(open(self.config.word2id, 'r'))}
        # label2id = {i : line.strip() for i, line in enumerate(open(self.config.label2id, 'r'))}
        infer2id = {i : line.strip() for i, line in enumerate(open(self.config.infer2id, 'r'))}
        infer2id_ent = {i : line.strip() for i, line in enumerate(open(self.config.infer2id_ent, 'r'))}
        true_seqs = []
        pred_seqs = []
        true_seqs_ent = []
        pred_seqs_ent = []

        if hasattr(self.config, 'save_eval_to_file') and self.config.save_eval_to_file:
            fout = open(self.config.eval_op_path + '.' + name, 'w')
        for res in results:
            if res['tokens'] is None:
                break
            word = map(lambda x: word2id.get(x, '<UNK>'), res['tokens'])
            tags = map(lambda x: infer2id.get(x, 'O'), res['tags'])
            pred = map(lambda x: infer2id.get(x, 'O'), res['pred'])
            for i in range(len(word)):
                if res['tokens'][i] == 0:
                    true_seqs += (tags[:i]+['O'])
                    pred_seqs += (pred[:i]+['O'])
                    break
                if hasattr(self.config, 'save_eval_to_file') and self.config.save_eval_to_file:
                    fout.write(word[i] + ' ')
                    # if word[i] == '<UNK>':
                    #     fout.write('O O\n')
                    # else:
                    fout.write(tags[i] + ' ')
                    fout.write(pred[i] + '\n')
            if hasattr(self.config, 'save_eval_to_file') and self.config.save_eval_to_file:
                fout.write('\n')

            # Chunking task only!
            if hasattr(self.config, 'chunking') and self.config.chunking:
                tags = map(lambda x: infer2id_ent.get(x, 'O'), res['tags'])
                pred = map(lambda x: infer2id_ent.get(x, 'O'), res['pred'])
                for i in range(len(word)):
                    if res['tokens'][i] == 0:
                        true_seqs_ent += (tags[:i] + ['O'])
                        pred_seqs_ent += (pred[:i] + ['O'])
                        break

        verbose = self.config.display_eval
        acc, prec, rec, f1_op = entity_f1(true_seqs_ent, pred_seqs_ent, verbose=verbose)
        fetch_results['Ent_f1'] = f1_op / 100
        fetch_results['accuracy'] = acc / 100
        fetch_results['precision'] = prec / 100
        fetch_results['recall'] = rec / 100

        return fetch_results['Ent_f1']

    def eval(self, is_in_train=False):
        if not is_in_train:
            self.build_graph(mode=tf.estimator.ModeKeys.EVAL, use_best=True)

        logger.info('Start evaling ...')
        score = 0
        watch_start = time.time()
        for dataset_config in self.eval_configs:
            self.reset_metric()
            fetch_result = None
            step = 0
            if hasattr(self.config, 'eval_with_input'):
                results = []
                for batch in self.datasets[dataset_config.name].make_iter(self.config.batch_size):
                    fetch_result = self.feedforward(
                        batch=batch,
                        mode=tf.estimator.ModeKeys.EVAL,
                        name=dataset_config.name,
                        with_input=True,
                    )
                    step += 1
                    for single_result in self.iter_fetch_data(fetch_result['predictions']):
                        results.append(single_result)

                score = self.update_fetch_result(dataset_config.name, fetch_result, results)
            else:
                for batch in self.datasets[dataset_config.name].make_iter(self.config.batch_size):
                    fetch_result = self.feedforward(
                        batch=batch,
                        mode=tf.estimator.ModeKeys.EVAL,
                        name=dataset_config.name,
                    )
                    step += 1

                self.update_fetch_result(dataset_config.name, fetch_result)

            self.log_result(
                name=dataset_config.name,
                speed=time.time() - watch_start,
                step=step,
                fetch_result=fetch_result,
            )
            watch_start = time.time()

            if is_in_train and hasattr(self.config, 'save_best_result'):
                self.update_best_result(dataset_config.name, fetch_result)

        if is_in_train:
            self.reset_metric()

            # skip normal eval save
            if hasattr(self.config, 'skip_eval_save') and self.config.skip_eval_save:
                return score
            self._save_checkpoint()
        return score

    def infer(self):
        self.build_graph(mode=tf.estimator.ModeKeys.PREDICT, use_best=True)
        logger.info('Start infering ...')
        results = []
        for batch in self.datasets[self.infer_name].make_iter(self.config.batch_size):
            fetch_result = self.feedforward(
                batch=batch,
                mode=tf.estimator.ModeKeys.PREDICT,
                name=self.infer_name,
                with_input=True,
            )
            for single_result in self.iter_fetch_data(fetch_result['predictions']):
                results.append(single_result)

        word2id = {i : line.strip() for i, line in enumerate(open(self.config.word2id, 'r'))}
        infer2id = {i : line.strip() for i, line in enumerate(open(self.config.infer2id, 'r'))}
        true_seqs = []
        pred_seqs = []

        with open(self.config.infer_op_path, 'w') as fout:
            for res in results:
                if res['tokens'] is None:
                    break
                word = map(lambda x: word2id.get(x, '<UNK>'), res['tokens'])
                pred = map(lambda x: infer2id.get(x, 'O'), res['pred'])
                for i in range(len(word)):
                    if res['tokens'][i] == 0:
                        pred_seqs += pred[:i]
                        break
                    fout.write(word[i]+' ')
                    fout.write(pred[i]+' \n')
                fout.write('\n')

    def _restore_checkpoint(self, use_best=False):
        if not use_best:
            checkpoint_dir = self.config.checkpoint_dir
        else:
            checkpoint_dir = self.config.best_checkpoint_dir
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        if checkpoint_file:
            logger.info('Restoring from checkpoint file %s' % checkpoint_file)
            self.saver.restore(self.sess, checkpoint_file)
        else:
            logger.info('Can not find any checkpoint file, start from zero!')

    def _save_checkpoint(self, name=None, use_best=False):
        if not use_best:
            checkpoint_dir = self.config.checkpoint_dir
        else:
            checkpoint_dir = self.config.best_checkpoint_dir
        logger.info('Saving to checkpoint file %s-%d' % (checkpoint_dir, self.global_step))
        if not name:
            self.saver.save(self.sess, os.path.join(checkpoint_dir, 'model.cpkt'), global_step=self.global_step)
        else:
            self.saver.save(self.sess, os.path.join(checkpoint_dir, name), global_step=self.global_step)

def main():
    pass


if __name__ == '__main__':
    main()
