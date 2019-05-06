#/usr/bin/env python
#-*- coding: utf-8 -*-


import os
import sys
import tensorflow as tf
#import numpy as np

import lib.layers as layers
import lib.utils.config
import model_base

class SeqLabellingModel(model_base.ModelBase):
    """Bi-LSTM-CRF Sequence Labelling Model"""
    def __init__(self, config):
        super(SeqLabellingModel, self).__init__(config)

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

        if config.use_crf:
            self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes, self.trans_params, self.seq_len)
        else:
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

        # Network Layers
        lstm_layer = layers.BiLSTMLayer(config.hidden_size, name='lstm')
        seq_fc_layer = layers.SeqFCLayer(config.hidden_size*2, config.n_classes, name='seq_fc')

        # Build graph
        tokens = inputs['tokens']
        label = tf.cast(inputs['tags'], tf.int32)
        nwords = tf.cast(inputs['nwords'], tf.int32)
        self.labels = label
        n_classes = self.config.n_classes

        emb = emb_layer(tokens)

        # pretrain_emb = np.empty([23626, config.emb_size])
        # scale = np.sqrt(3.0 / config.emb_size)
        # for index in range(23626):
        #     pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, config.emb_size])
        
        # init = tf.constant_initializer(pretrain_emb)
        # U = tf.get_variable('U', shape=[23626, config.emb_size], initializer=init)
        # with tf.variable_scope("U", reuse=tf.AUTO_REUSE):
        #         U = tf.from_numpy(pretrain_emb, dtype=tf.float32)
        # emb = tf.nn.embedding_lookup(U, tokens)

        # glove = np.load('./data/ner_BIOES/glove.npz')['embeddings']  # np.array
        # variable = np.vstack(glove)
        # variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
        # emb = tf.nn.embedding_lookup(variable, tokens)
        # print emb
        emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        h, _ = lstm_layer(emb, x_length=nwords)
        h = tf.layers.dropout(h, rate=config.dropout_rate, training=training)

        # seq_logits = seq_fc_layer(h)
        seq_logits = tf.layers.dense(h, n_classes)
        # seq_logits = tf.reshape(seq_logits, [-1, self.config.max_length, self.config.n_classes])

        weights = tf.cast(tf.not_equal(tokens, 0), dtype=tf.float32)
        if config.use_crf:
            crf_params = tf.get_variable("crf", [n_classes, n_classes], dtype=tf.float32)
            predicted, _ = tf.contrib.crf.crf_decode(seq_logits, crf_params, nwords)
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(seq_logits, label, nwords, crf_params)

            self.loss_op = tf.reduce_mean(-log_likelihood)
            self.trans_params = trans_params

            # seq_len = tf.constant(np.full(config.batch_size, config.max_length, dtype=np.int64))
            # log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(seq_logits, label, seq_len)
            # self.trans_params = trans_params
            # self.loss_op = -tf.reduce_mean(log_likelihood)
            # predicted, _ = tf.contrib.crf.crf_decode(seq_logits, trans_params, seq_len)
            self.infer_op = tf.one_hot(predicted, depth=n_classes, on_value=1.0, off_value=0.0, axis=-1)
        else:
            self.loss_op = tf.contrib.seq2seq.sequence_loss(seq_logits, label, weights, average_across_timesteps=True, \
                    average_across_batch=True)
            self.infer_op = seq_logits

        self.logits = seq_logits
        self.logits_op = seq_logits
        
        if config.use_crf:
            metric_layer = lib.layers.NERMetricLayer()
            self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes, weights=weights, trans_params=self.trans_params, nwords=nwords)
        else:
            metric_layer = lib.layers.MultiClassificationMetricLayer()
            self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes, weights=weights)

class RegionSeqLabellingModel(model_base.ModelBase):
    """Region Embedding Sequence Labelling Model"""
    def __init__(self, config):
        super(RegionSeqLabellingModel, self).__init__(config)

    def build_graph(self, inputs, mode):
        config = self.config
        # print inputs
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Embedding Layer
        if not config.use_word_pretrain_emb:
            emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='emb')
        else:
            emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size, config.emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb')

        # Network Layers
        seq_fc_layer = layers.SeqFCLayer(config.emb_size, config.n_classes, name='seq_fc')

        # Build graph
        tokens = inputs['tokens']
        label = tf.cast(inputs['tags'], tf.int32)
        nwords = tf.cast(inputs['nwords'], tf.int32)
        self.labels = label
        n_classes = self.config.n_classes

        # emb = emb_layer(tokens)
        # # emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        # # Context matrix
        # U = tf.get_variable('region_embedding_U',\
        #      shape=[self.config.vocab_size, self.config.region_size, self.config.emb_size], dtype=tf.float32, trainable=True)
        # context_units = tf.nn.embedding_lookup(U, tokens)
        # region_radius = self.config.region_size / 2    

        char_emb = np.load(config.word2vec_dict)['embeddings']
        # char_emb = np.random.randn(config.vocab_size, config.emb_size)
        # char_emb = tf.cast(char_emb, dtype=tf.float32)
        char_emb = tf.get_variable('char_emb', initializer=char_emb, dtype=tf.float32, trainable=config.word_emb_finetune)
        emb = tf.nn.embedding_lookup(char_emb, tokens)
        # emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        # Context matrix
        U = np.load(config.word2context_dict)['embeddings']
        # U = np.random.randn(config.vocab_size, config.region_size, config.emb_size)
        # U = tf.cast(U, dtype=tf.float32)
        U = tf.get_variable('U', initializer=U, dtype=tf.float32, trainable=config.context_emb_finetune)
        context_units = tf.nn.embedding_lookup(U, tokens)
        region_radius = self.config.region_size / 2

        if not self.config.context_word:
            # Word-Context Embedding
            paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0]])
            paded_seq_emb = tf.pad(emb, paddings, "CONSTANT")
            region_aligned_emb = tf.map_fn(lambda i: paded_seq_emb[:, i: i + self.config.region_size, ...],
                          tf.range(self.config.max_length), dtype=tf.float32)
            region_aligned_emb = tf.transpose(region_aligned_emb, perm=[1, 0, 2, 3])
            projected_emb = region_aligned_emb * context_units
        else:
            # Context-Word Embedding
            paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0], [0, 0]])
            paded_context_units = tf.pad(context_units, paddings, "CONSTANT")
            region_aligned_unit = tf.map_fn(
                lambda i: tf.stack(
                [paded_context_units[:,i+k,self.config.region_size-1-k,:] for k in range(self.config.region_size)],axis=-1),
                tf.range(self.config.max_length), dtype=tf.float32)
            region_aligned_unit = tf.transpose(region_aligned_unit, perm=[1, 0, 3, 2])
            emb = tf.expand_dims(emb, 2)
            projected_emb = region_aligned_unit * emb

        # Max Pooling
        h = tf.reduce_max(projected_emb, axis=2)

        # seq_logits = seq_fc_layer(h)
        for i in range(5):
            h = tf.layers.dense(h, config.emb_size, activation=tf.nn.relu, name='fc_'+str(i))
        seq_logits = tf.layers.dense(h, n_classes, name='seq_fc')

        weights = tf.cast(tf.not_equal(tokens, 0), dtype=tf.float32)
        if config.use_crf:
            crf_params = tf.get_variable("crf", [n_classes, n_classes], dtype=tf.float32)
            predicted, _ = tf.contrib.crf.crf_decode(seq_logits, crf_params, nwords)
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(seq_logits, label, nwords, crf_params)

            self.loss_op = tf.reduce_mean(-log_likelihood)
            self.trans_params = trans_params
            self.infer_op = tf.one_hot(predicted, depth=n_classes, on_value=1.0, off_value=0.0, axis=-1)
        else:
            self.loss_op = tf.contrib.seq2seq.sequence_loss(seq_logits, label, weights, average_across_timesteps=True, \
                    average_across_batch=True)
            self.infer_op = seq_logits

        self.logits = seq_logits
        self.logits_op = seq_logits
        
        if config.use_crf:
            metric_layer = lib.layers.NERMetricLayer()
            self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes, weights=weights, trans_params=self.trans_params, nwords=nwords)
        else:
            metric_layer = lib.layers.MultiClassificationMetricLayer()
            self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes, weights=weights)


class RankRegionSeqLabellingModel(model_base.ModelBase):
    """Region Embedding + Pretrain Embedding Sequence Labelling Model"""
    def __init__(self, config):
        super(RankRegionSeqLabellingModel, self).__init__(config)

    def build_graph(self, inputs, mode):
        config = self.config
        # print inputs
        training = (mode == tf.estimator.ModeKeys.TRAIN)

        # Network Layers
        # seq_fc_layer = layers.SeqFCLayer(config.fc_in, config.n_classes, name='seq_fc')

        # Build graph
        tokens = inputs['tokens_rand']
        label = tf.cast(inputs['tags'], tf.int32)
        nwords = tf.cast(inputs['nwords'], tf.int32)
        self.labels = label
        n_classes = self.config.n_classes

        emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='emb')
        emb = emb_layer(tokens)
        emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        # Context matrix
        K = tf.get_variable('K',\
             shape=[self.config.width, self.config.region_size * self.config.emb_size], dtype=tf.float32, trainable=True)
        alpha = tf.get_variable('alpha',\
            shape=[self.config.vocab_size, self.config.width], dtype=tf.float32, trainable=True)
        context_weights = tf.nn.embedding_lookup(alpha, tokens, name='context_weights')
        context_weights = tf.reshape(context_weights, [-1, self.config.width])
        # print context_weights
        # print K
        context_units = tf.matmul(context_weights, K, name='context_units')
        context_units = tf.reshape(context_units, [-1, self.config.max_length, self.config.region_size, self.config.emb_size])
        print context_units
        region_radius = self.config.region_size / 2

        if not self.config.context_word:
            # Word-Context Embedding
            paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0]])
            paded_seq_emb = tf.pad(emb, paddings, "CONSTANT")
            region_aligned_emb = tf.map_fn(lambda i: paded_seq_emb[:, i: i + self.config.region_size, ...],
                          tf.range(self.config.max_length), dtype=tf.float32)
            region_aligned_emb = tf.transpose(region_aligned_emb, perm=[1, 0, 2, 3])
            projected_emb = region_aligned_emb * context_units
        else:
            # Context-Word Embedding
            paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0], [0, 0]])
            paded_context_units = tf.pad(context_units, paddings, "CONSTANT")
            region_aligned_unit = tf.map_fn(
                lambda i: tf.stack(
                [paded_context_units[:,i+k,self.config.region_size-1-k,:] for k in range(self.config.region_size)],axis=-1),
                tf.range(self.config.max_length), dtype=tf.float32)
            region_aligned_unit = tf.transpose(region_aligned_unit, perm=[1, 0, 3, 2])
            emb = tf.expand_dims(emb, 2)
            projected_emb = region_aligned_unit * emb

        # Max Pooling
        region_merge_fn = tf.reduce_max
        h = tf.reduce_max(projected_emb, axis=2)

        # Embedding Layer
        if config.concat:
            if not config.use_word_pretrain_emb:
                emb_layer1 = layers.EmbeddingLayer(config.vocab_size, config.emb1_size, name='emb1')
            else:
                emb_layer1 = layers.InitializedEmbeddingLayer(config.vocab_size, config.emb1_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb1')
            tokens_emb = inputs['tokens']
            emb1 = emb_layer1(tokens_emb)
            h = tf.concat([h,emb1], axis=-1)
            print h

        # seq_logits = seq_fc_layer(h)
        seq_logits = tf.layers.dense(h, n_classes)
        # print seq_logits
        # seq_logits = tf.reshape(seq_logits, [-1, self.config.max_length, self.config.n_classes])

        weights = tf.cast(tf.not_equal(tokens, 0), dtype=tf.float32)
        if config.use_crf:
            crf_params = tf.get_variable("crf", [n_classes, n_classes], dtype=tf.float32)
            predicted, _ = tf.contrib.crf.crf_decode(seq_logits, crf_params, nwords)
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(seq_logits, label, nwords, crf_params)

            self.loss_op = tf.reduce_mean(-log_likelihood)
            self.trans_params = trans_params

            # seq_len = tf.constant(np.full(config.batch_size, config.max_length, dtype=np.int64))
            # log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(seq_logits, label, seq_len)
            # self.trans_params = trans_params
            # self.loss_op = -tf.reduce_mean(log_likelihood)
            # predicted, _ = tf.contrib.crf.crf_decode(seq_logits, trans_params, seq_len)
            self.infer_op = tf.one_hot(predicted, depth=n_classes, on_value=1.0, off_value=0.0, axis=-1)
        else:
            self.loss_op = tf.contrib.seq2seq.sequence_loss(seq_logits, label, weights, average_across_timesteps=True, \
                    average_across_batch=True)
            self.infer_op = seq_logits

        self.logits = seq_logits
        self.logits_op = seq_logits
        
        if config.use_crf:
            metric_layer = lib.layers.NERMetricLayer()
            self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes, weights=weights, trans_params=self.trans_params, nwords=nwords)
        else:
            metric_layer = lib.layers.MultiClassificationMetricLayer()
            self.metric = metric_layer(self.logits_op, self.labels, self.config.n_classes, weights=weights)

def main():
    model = TestModel(conf)

if '__main__' == __name__:
    main()

