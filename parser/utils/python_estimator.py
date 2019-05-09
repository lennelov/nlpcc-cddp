#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import datetime
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from utils import get_dataset
import traceback

logger = logging.getLogger(__name__)


class PythonEstimator(object):
    """A Python Estimator which is more flexible than tf.Estimator"""

    def __init__(self, conf, model):
        self.config = conf
        self.model = model

        if not hasattr(self.config, 'log_every_n_steps'):
            self.config.add('log_every_n_steps', 100)

        if not hasattr(self.config, 'max_to_keep'):
            self.config.add('max_to_keep', 50)

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        self.summaries_dir = os.path.join(self.config.checkpoint_dir, 'summary')
        if not os.path.exists(self.summaries_dir):
            os.makedirs(self.summaries_dir)

        # ----------- check and reformat dataset config -----------
        self.dataset_configs = []
        self.eval_configs = []

        assert hasattr(self.config, 'dev_dataset') or \
               hasattr(self.config, 'eval_datasets') or \
               hasattr(self.config, 'train_dataset') or \
               hasattr(self.config, 'infer_dataset')

        if hasattr(self.config, 'dev_dataset'):
            self.dataset_configs.append(self.config.dev_dataset)
            self.eval_configs.append(self.config.dev_dataset)

        if hasattr(self.config, 'eval_datasets'):
            self.dataset_configs += self.config.eval_datasets
            self.eval_configs += self.config.eval_datasets

        if hasattr(self.config, 'train_dataset'):
            if not hasattr(self.config.train_dataset, 'name'):
                self.config.train_dataset.add('name', 'train')
            self.train_name = self.config.train_dataset.name
            self.dataset_configs += [self.config.train_dataset]

        if hasattr(self.config, 'infer_dataset'):
            if not hasattr(self.config.infer_dataset, 'name'):
                self.config.infer_dataset.add('name', 'infer')
            self.infer_name = self.config.infer_dataset.name
            self.dataset_configs += [self.config.infer_dataset]

        # ----------- build dataset config -----------
        # name can be same
        for i, dataset_config in enumerate(self.dataset_configs):
            assert 'Python' in dataset_config.type, 'dataset type error'
            if not hasattr(dataset_config, 'name'):
                dataset_config.add('name', 'eval' + str(i))

        # ----------- build datasets, writers and model_inputs -----------
        self.datasets = {}
        self.writers = {}
        self.model_inputs = {}
        for i, dataset_config in enumerate(self.dataset_configs):
            logger.info('Building ' + dataset_config.name + '...')
            self.datasets[dataset_config.name] = get_dataset(dataset_config)
            self.datasets[dataset_config.name].build(self.model_inputs)
            self.model_inputs.update(self.datasets[dataset_config.name].inputs)
            self.writers[dataset_config.name] = tf.summary.FileWriter(
                os.path.join(self.summaries_dir, dataset_config.name)
            )
        # self.eval_datasets = [self.datasets[x] for x in self.eval_datasets]

        self.config.best_checkpoint_dir = self.config.checkpoint_dir + '/best'

    def set_eval_and_summary(self):
        self.eval_and_summary = []
        for key, value in self.estimator_spec.eval_metric_ops.items():
            self.eval_and_summary.append(key)

    def make_fetch_dict(self, mode):
        fetch_dict = {}
        if mode == 'EXPORT':
            fetch_dict['predictions'] = self.estimator_spec.predictions
        elif mode == tf.estimator.ModeKeys.PREDICT:
            fetch_dict['predictions'] = self.estimator_spec.predictions
        else:
            for k, v in self.estimator_spec.eval_metric_ops.items():
                fetch_dict[k] = v[1]
            fetch_dict['predictions'] = self.estimator_spec.predictions
            if mode == tf.estimator.ModeKeys.TRAIN:
                fetch_dict['train_op'] = self.estimator_spec.train_op
                fetch_dict['global_step'] = self.global_step
                fetch_dict['loss'] = self.estimator_spec.loss
        return fetch_dict

    def build_graph(self, mode, use_best=False):
        self.action_mode = mode
        self.fetch_dict = {}
        self.add_estimator_inputs(mode)
        # if mode == 'EXPORT':
        #     self.estimator_spec = self.model.model_fn(self.model_inputs, mode=tf.estimator.ModeKeys.PREDICT)
        #     self.fetch_dict[mode] = self.make_fetch_dict(mode)
        # else:
        self.estimator_spec = self.model.model_fn(self.model_inputs, mode=mode)
        self.set_eval_and_summary()
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.global_step = tf.train.get_global_step()

            self.global_epoch = 0
            self.fetch_dict[tf.estimator.ModeKeys.EVAL] = self.make_fetch_dict(tf.estimator.ModeKeys.EVAL)
        self.fetch_dict[mode] = self.make_fetch_dict(mode)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.config.max_to_keep)
        logger.info('Start session ...')

        self.sess = tf.Session()
        if hasattr(self.config, 'debug') and self.config.debug.enabled:
            logger.debug('Listing all variables in graph:')
            for v in tf.get_default_graph().as_graph_def().node:
                logger.debug(v)

            assert self.config.debug.type in ['LocalCLIDebugWrapperSession', 'TensorBoardDebugWrapperSession'], \
                    'unsupported debug wrapper session!'

            if self.config.debug.type == 'TensorBoardDebugWrapperSession':
                self.sess = TensorBoardDebugWrapperSession(self.sess)

            if self.config.debug.type =='LocalCLIDebugWrapperSession':
                self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
            logger.info('Debuging as %s' % type(self.sess))
            self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        writer = tf.summary.FileWriter(self.summaries_dir, self.sess.graph)
        writer.close()
        self._restore_checkpoint(use_best=use_best)

    def add_estimator_inputs(self, mode):
        if hasattr(self.config, 'dropout_keep_prob') and mode != 'EXPORT':
            self.model_inputs['dropout_keep_prob'] = tf.placeholder(tf.float32, name="dropout_keep_prob")
            logger.info('using dropout_keep_prob : %s', self.model_inputs['dropout_keep_prob'])

    def update_estimator_feed_dict(self, batch, mode=tf.estimator.ModeKeys.TRAIN, *args, **kwargs):
        if hasattr(self.config, 'dropout_keep_prob'):
            if mode == tf.estimator.ModeKeys.TRAIN:
                batch[self.model_inputs['dropout_keep_prob']] = self.config.dropout_keep_prob
            else:
                batch[self.model_inputs['dropout_keep_prob']] = 1

    def feedforward(self, batch, mode, name, with_input=False):

        # update input data i.e. dropout rate
        self.update_estimator_feed_dict(batch, mode)
        fetch_dict = {}
        fetch_dict.update(self.fetch_dict[mode])

        try:
            fetch_result = self.sess.run(fetch_dict, feed_dict=batch)
        except ValueError as e:
            for k, v in fetch_dict.items():
                logger.error('fetch dict[%s] = %s', k, v)
            for k, v in batch.items():
                logger.error('Feed dict[%s] = %s', k, np.array(v).shape)
            traceback.print_exc()
            raise e

        # update global step
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.global_step = fetch_result['global_step']

        # put input into output
        if with_input:
            for k, v in batch.items():
                k = k.name
                k = k.replace(':0', '')
                k = k.replace('_placeholder', '')
                fetch_result['predictions'][k] = v
        return fetch_result

    def log_result(self, name, speed, step, fetch_result):
        if name == self.train_name:
            common_output = '[%s][Epoch:%s][Step:%s][%.1f s][%.1f step/s]' % (
                name, self.global_epoch, self.global_step, speed, step / speed)
            eval_output = ''.join(['[%s:%s]' % (k, v) for k, v in fetch_result.items() if
                                   k not in ['train_op', 'global_step', 'summary', 'predictions']])
        else:
            common_output = '[%s][%.1f s][%.1f step/s]' % (name, speed, step / speed)
            eval_output = ''.join(['{%s:%s}' % (k, v) for k, v in fetch_result.items() if
                                   (k not in ['train_op', 'global_step', 'summary',
                                              'predictions'] and 'best' not in k)])
        if self.action_mode == tf.estimator.ModeKeys.TRAIN:
            summary = tf.Summary()
            for k in self.eval_and_summary:
                if k in fetch_result:
                    summary.value.add(tag=k, simple_value=fetch_result[k])

            self.writers[name].add_summary(summary, self.global_step)

        output = common_output + "\t" + eval_output

        logger.info(output)

    def reset_metric(self):
        self.sess.run(tf.local_variables_initializer())

    def update_fetch_result(self, name, fetch_results, results=None):
        pass

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
            results = []
            for batch in self.datasets[dataset_config.name].make_iter(self.config.batch_size):
                fetch_result = self.feedforward(
                    batch=batch,
                    mode=tf.estimator.ModeKeys.EVAL,
                    name=dataset_config.name,
                )
                step += 1
                for single_result in self.iter_fetch_data(fetch_result['predictions']):
                    results.append(single_result)

            self.update_fetch_result(dataset_config.name, fetch_result)
            score = fetch_result['uas']

            self.log_result(
                name=dataset_config.name,
                speed=time.time() - watch_start,
                step=step,
                fetch_result=fetch_result,
            )
            watch_start = time.time()

            if hasattr(self.config, 'eval_to_file') and self.config.eval_to_file:
                word2id = {i : line.strip() for i, line in enumerate(open(self.config.word2id, 'r'))}
                pos2id = {i : line.strip() for i, line in enumerate(open(self.config.pos2id, 'r'))}

                fout = open(self.config.eval_op_path+'.'+dataset_config.name, 'w')
                for res in results:
                    if res['word'] is None:
                        break
                    idx = res['idx']
                    word = map(lambda x: word2id.get(x, '<UNK>'), res['word'])
                    upos = map(lambda x: pos2id.get(x, '<UNK>'), res['upos'])
                    xpos = map(lambda x: pos2id.get(x, '<UNK>'), res['xpos'])
                    pred = res['pred']
                    for i in range(len(word)):
                        if res['word'][i] == 0:
                            break
                        fout.write('\t'.join([str(idx[i]),'_',word[i],upos[i],xpos[i],'_',str(pred[i]),'_','_','_'])+'\n')
                    fout.write('\n')
                fout.close()

        if is_in_train:
            self.reset_metric()

            # skip normal eval save
            if hasattr(self.config, 'skip_eval_save') and self.config.skip_eval_save:
                return
            self._save_checkpoint()
        return score

    def do_something_when_epoch_over(self, epoch_time=None):
        pass

    def train(self, train_dataset=None):
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
                    print score, count, self.config.tolerance, count >= self.config.tolerance
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

                if hasattr(self.config, 'max_training_steps') and \
                        self.global_step > self.config.max_training_steps:
                    score = self.eval(is_in_train=True)
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

            self.do_something_when_epoch_over(time.time() - epoch_start_time)

            if training_finished:
                break

            logger.info('Epoch %s finished, %s elapsed.', self.global_epoch,
                        datetime.timedelta(seconds=time.time() - start_time))
            self.global_epoch += 1
        logger.info('Training finished, %s elapsed.', datetime.timedelta(seconds=time.time() - start_time))

    def iter_fetch_data(self, fetch_result):
        for i in range(self.config.batch_size):
            data_point = {}
            out_flag = False
            for k, v in fetch_result.items():
                if k in ['dropout_keep_prob']:
                    continue
                if i >= len(v):
                    out_flag = True
                    break
                data_point[k] = fetch_result[k][i]
            if out_flag:
                break
            yield data_point

    def infer(self, infer_dataset=None):
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
        pos2id = {i : line.strip() for i, line in enumerate(open(self.config.pos2id, 'r'))}

        fout = open(self.config.infer_op_path, 'w')
        for res in results:
            if res['word'] is None:
                break
            idx = res['idx']
            word = map(lambda x: word2id.get(x, '<UNK>'), res['word'])
            upos = map(lambda x: pos2id.get(x, '<UNK>'), res['upos'])
            xpos = map(lambda x: pos2id.get(x, '<UNK>'), res['xpos'])
            pred = res['pred']
            for i in range(len(word)):
                if res['word'][i] == 0:
                    break
                fout.write('\t'.join([str(idx[i]),'_',word[i],upos[i],xpos[i],'_',str(pred[i]),'_','_','_'])+'\n')
            fout.write('\n')
        fout.close()

    # def export_model(self):
    #     self.build_graph(mode='EXPORT')

    #     export_dir = os.path.join(self.config.checkpoint_dir, 'saved_model')
    #     outputs = self.estimator_spec.predictions

    #     for k, v in self.model_inputs.items():
    #         logger.info('inputs: [key: %s], [value: %s]', k, v)
    #     for k, v in outputs.items():
    #         logger.info('outputs: [key: %s], [value: %s]', k, v)

    #     tf.saved_model.simple_save(self.sess, export_dir, inputs=self.model_inputs, outputs=outputs)

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
