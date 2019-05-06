#/usr/bin/env python
#-*- coding: utf-8 -*-


import os
import sys
import tensorflow as tf
import numpy as np

import lib.layers as layers
import lib.utils.config
import model_base
from lstm_chunking_model import ChunkingModel

class RegionChunkingModel(ChunkingModel):
    """Region Embedding Chunking Model"""
    def __init__(self, config):
        super(RegionChunkingModel, self).__init__(config)

    def build_graph(self, inputs, mode):
        config = self.config
        # print inputs
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Embedding Layer
        if not config.use_word_pretrain_emb:
            emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='emb')
            gram_emb_layer = layers.EmbeddingLayer(config.gram_size, config.gram_emb_size, name='emb_g')
        else:
            emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size, config.emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb')
            gram_emb_layer = layers.InitializedEmbeddingLayer(config.gram_size, config.gram_emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb_g')

        # Build graph
        tokens = inputs['tokens']
        label = tf.cast(inputs['tags'], tf.int32)
        nwords = tf.cast(inputs['nwords'], tf.int32)
        grams = inputs['grams']
        self.labels = label
        n_classes = self.config.n_classes

        emb = emb_layer(tokens)
        emb_gram = gram_emb_layer(grams)
        grams_f = tf.concat([grams[:,-1:], grams[:,:-1]], axis=-1)
        emb_gram_f = gram_emb_layer(grams_f)

        emb = tf.concat([emb, emb_gram, emb_gram_f], axis=-1)
        emb = tf.layers.dropout(emb, rate=self.config.dropout_rate, training=training)

        # # Build graph
        # tokens = inputs['tokens']
        # label = tf.cast(inputs['tags'], tf.int32)
        max_length = config.max_length
        # nwords = tf.cast(inputs['nwords'], tf.int32)
        # # nwords = tf.clip_by_value(tf.cast(inputs['nwords'], tf.int32), 0, config.max_length)
        # # max_length = tf.reduce_max(nwords)
        # # nwords = tf.cast(inputs['nwords'], tf.int32)
        # self.labels = label
        # n_classes = self.config.n_classes

        # emb = emb_layer(tokens)
        # emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        # Context matrix
        U = tf.get_variable('region_embedding_U',\
             shape=[self.config.vocab_size, self.config.region_size, 2*self.config.gram_emb_size+self.config.emb_size], dtype=tf.float32, trainable=True)
        context_units = tf.nn.embedding_lookup(U, tokens)
        region_radius = self.config.region_size / 2    

        if not self.config.context_word:
            # Word-Context Embedding
            paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0]])
            paded_seq_emb = tf.pad(emb, paddings, "CONSTANT")
            region_aligned_emb = tf.map_fn(lambda i: paded_seq_emb[:, i: i + self.config.region_size, ...],
                          tf.range(max_length), dtype=tf.float32)
            region_aligned_emb = tf.transpose(region_aligned_emb, perm=[1, 0, 2, 3])
            projected_emb = region_aligned_emb * context_units
        else:
            # Context-Word Embedding
            paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0], [0, 0]])
            paded_context_units = tf.pad(context_units, paddings, "CONSTANT")
            region_aligned_unit = tf.map_fn(
                lambda i: tf.stack(
                [paded_context_units[:,i+k,self.config.region_size-1-k,:] for k in range(self.config.region_size)],axis=-1),
                tf.range(max_length), dtype=tf.float32)
            region_aligned_unit = tf.transpose(region_aligned_unit, perm=[1, 0, 3, 2])
            emb = tf.expand_dims(emb, 2)
            projected_emb = region_aligned_unit * emb

        projected_emb = tf.layers.dropout(projected_emb, rate=self.config.dropout_rate, training=training)
        # Max Pooling
        h = tf.reduce_max(projected_emb, axis=2)

        if hasattr(self.config, 'pre_classification') and self.config.pre_classification:
            # rep_size = (2*self.config.gram_emb_size+self.config.emb_size) / 24
            rep = tf.layers.dense(h, self.config.rep_size)
            for idx in range(1, self.config.width+1):
                h_f = tf.concat([tf.multiply(rep[:,:,-idx:],0),rep[:,:,:-idx]],axis=-1)
                h_b = tf.concat([rep[:,:,idx:],tf.multiply(rep[:,:,:idx],0)],axis=-1)
                h = tf.concat([h,h_f,h_b],axis=-1)
            seq_fc_layer = layers.SeqFCLayer(self.config.rep_size*2*self.config.width + 2*self.config.gram_emb_size+self.config.emb_size, config.n_classes, name='seq_fc')
        else:
            seq_fc_layer = layers.SeqFCLayer(2*self.config.gram_emb_size+self.config.emb_size, config.n_classes, name='seq_fc')
        
        seq_logits = seq_fc_layer(h)
        # seq_logits = tf.layers.dense(h, n_classes)

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
            if hasattr(self.config, 'pre_classification') and self.config.pre_classification:
                # logits_rep = rep
                rep = tf.math.sigmoid(rep)
                logits_rep = layers.SeqFCLayer(self.config.rep_size, config.n_classes, name='rep_seq_fc')(rep)
                self.loss_op += tf.contrib.seq2seq.sequence_loss(logits_rep, label, weights, average_across_timesteps=True, \
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

class PretrainRegionChunkingModel(RegionChunkingModel):
    """Pretrained Region Embedding Chunking Model"""
    def __init__(self, config):
        super(PretrainRegionChunkingModel, self).__init__(config)

    def build_graph(self, inputs, mode):
        config = self.config
        # print inputs
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        max_length = config.max_length
        # # Embedding Layer
        # if not config.use_word_pretrain_emb:
        #     emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='emb')
        # else:
        #     emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size, config.emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb')

        # Build graph
        tokens = inputs['tokens']
        label = tf.cast(inputs['tags'], tf.int32)
        nwords = tf.clip_by_value(tf.cast(inputs['nwords'], tf.int32), 0, max_length)

        self.labels = label
        n_classes = self.config.n_classes

        char_emb = np.load(config.word2vec_dict)['embeddings']
        # char_emb = np.random.randn(config.vocab_size, config.emb_size)
        # char_emb = tf.cast(char_emb, dtype=tf.float32)
        char_emb = tf.get_variable('char_emb', initializer=char_emb, dtype=tf.float32, trainable=config.word_emb_finetune)
        emb = tf.nn.embedding_lookup(char_emb, tokens)
        emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

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
                          tf.range(max_length), dtype=tf.float32)
            region_aligned_emb = tf.transpose(region_aligned_emb, perm=[1, 0, 2, 3])
            projected_emb = region_aligned_emb * context_units
        else:
            # Context-Word Embedding
            paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0], [0, 0]])
            paded_context_units = tf.pad(context_units, paddings, "CONSTANT")
            region_aligned_unit = tf.map_fn(
                lambda i: tf.stack(
                [paded_context_units[:,i+k,self.config.region_size-1-k,:] for k in range(self.config.region_size)],axis=-1),
                tf.range(max_length), dtype=tf.float32)
            region_aligned_unit = tf.transpose(region_aligned_unit, perm=[1, 0, 3, 2])
            emb = tf.expand_dims(emb, 2)
            projected_emb = region_aligned_unit * emb

        # Max Pooling
        h = tf.reduce_max(projected_emb, axis=2)

        # h = tf.transpose(h, perm=[1, 0, 2])
        # lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        # lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(config.hidden_size)
        # lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        # with tf.variable_scope('lstm1'):
        #     output_fw, _ = lstm_cell_fw(h, dtype=tf.float32, sequence_length=nwords)
        #     output_fw = tf.layers.dropout(output_fw, rate=config.dropout_rate, training=training)
        # with tf.variable_scope('lstm2'):
        #     output_bw, _ = lstm_cell_bw(output_fw, dtype=tf.float32, sequence_length=nwords)
        #     output_bw = tf.layers.dropout(output_bw, rate=config.dropout_rate, training=training)
        # # output = tf.concat([output_fw, output_bw], axis=-1)
        # output = output_bw
        # output = tf.transpose(output, perm=[1, 0, 2])
        
        # h = tf.concat(output, axis=2)

        for i in range(5):
            h = tf.layers.dense(h, config.emb_size, activation=tf.nn.relu, name='fc_'+str(i))
        # seq_logits = seq_fc_layer(h)
        seq_logits = tf.layers.dense(h, n_classes, name='fc')

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


class MultiRegionChunkingModel(ChunkingModel):
    """Region Embedding Chunking Model"""
    def __init__(self, config):
        super(MultiRegionChunkingModel, self).__init__(config)

    def build_graph(self, inputs, mode):
        config = self.config
        # print inputs
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        # Embedding Layer
        if not config.use_word_pretrain_emb:
            emb_layer = layers.EmbeddingLayer(config.vocab_size, config.emb_size, name='emb')
            gram_emb_layer = layers.EmbeddingLayer(config.gram_size, config.gram_emb_size, name='emb_g')
        else:
            emb_layer = layers.InitializedEmbeddingLayer(config.vocab_size, config.emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb')
            gram_emb_layer = layers.InitializedEmbeddingLayer(config.gram_size, config.gram_emb_size, config.word2vec_dict, trainable=config.word_emb_finetune, name='emb_g')

        # Build graph
        tokens = inputs['tokens']
        label = tf.cast(inputs['tags'], tf.int32)
        nwords = tf.cast(inputs['nwords'], tf.int32)
        grams = inputs['grams']
        self.labels = label
        n_classes = self.config.n_classes

        emb = emb_layer(tokens)
        emb_gram = gram_emb_layer(grams)
        grams_f = tf.concat([grams[:,-1:], grams[:,:-1]], axis=-1)
        emb_gram_f = gram_emb_layer(grams_f)

        emb = tf.concat([emb, emb_gram, emb_gram_f], axis=-1)
        emb = tf.layers.dropout(emb, rate=0.25, training=training)

        # # Build graph
        # tokens = inputs['tokens']
        # label = tf.cast(inputs['tags'], tf.int32)
        max_length = config.max_length
        # nwords = tf.cast(inputs['nwords'], tf.int32)
        # # nwords = tf.clip_by_value(tf.cast(inputs['nwords'], tf.int32), 0, config.max_length)
        # # max_length = tf.reduce_max(nwords)
        # # nwords = tf.cast(inputs['nwords'], tf.int32)
        # self.labels = label
        # n_classes = self.config.n_classes

        # emb = emb_layer(tokens)
        # emb = tf.layers.dropout(emb, rate=config.dropout_rate, training=training)

        region_s = [3, 7, 13]
        h = []
        if self.config.context_word:
            emb = tf.expand_dims(emb, 2)
        if hasattr(self.config, 'share') and self.config.share:
            U_total = tf.get_variable('region_embedding_U',\
                     shape=[self.config.vocab_size, max(region_s), 2*self.config.gram_emb_size+self.config.emb_size], dtype=tf.float32, trainable=True)
        for idx, s in enumerate(region_s):
            self.config.region_size = s
            # Context matrix
            shft = (max(region_s) - s) / 2
            if hasattr(self.config, 'share') and self.config.share:
                U = U_total[:,shft:max(region_s)-shft,:]
            else:
                U = tf.get_variable('region_embedding_U'+str(idx),\
                     shape=[self.config.vocab_size, s, 2*self.config.gram_emb_size+self.config.emb_size], dtype=tf.float32, trainable=True)
            context_units = tf.nn.embedding_lookup(U, tokens)
            region_radius = self.config.region_size / 2    

            if not self.config.context_word:
                # Word-Context Embedding
                paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0]])
                paded_seq_emb = tf.pad(emb, paddings, "CONSTANT")
                region_aligned_emb = tf.map_fn(lambda i: paded_seq_emb[:, i: i + self.config.region_size, ...],
                              tf.range(max_length), dtype=tf.float32)
                region_aligned_emb = tf.transpose(region_aligned_emb, perm=[1, 0, 2, 3])
                projected_emb = region_aligned_emb * context_units
            else:
                # Context-Word Embedding
                paddings = tf.constant([[0, 0], [region_radius, region_radius], [0, 0], [0, 0]])
                paded_context_units = tf.pad(context_units, paddings, "CONSTANT")
                region_aligned_unit = tf.map_fn(
                    lambda i: tf.stack(
                    [paded_context_units[:,i+k,self.config.region_size-1-k,:] for k in range(self.config.region_size)],axis=-1),
                    tf.range(max_length), dtype=tf.float32)
                region_aligned_unit = tf.transpose(region_aligned_unit, perm=[1, 0, 3, 2])
                projected_emb = region_aligned_unit * emb

            # Max Pooling
            h.append(tf.reduce_max(projected_emb, axis=2))

        h = tf.concat(h, axis=-1)

        if hasattr(self.config, 'pre_classification') and self.config.pre_classification:
            # rep_size = (2*self.config.gram_emb_size+self.config.emb_size) / 24
            rep = tf.layers.dense(h, self.config.rep_size)
            for idx in range(1, self.config.width+1):
                h_f = tf.concat([tf.multiply(rep[:,:,-idx:],0),rep[:,:,:-idx]],axis=-1)
                h_b = tf.concat([rep[:,:,idx:],tf.multiply(rep[:,:,:idx],0)],axis=-1)
                h = tf.concat([h,h_f,h_b],axis=-1)
            seq_fc_layer = layers.SeqFCLayer(self.config.rep_size*2*self.config.width + len(region_s)*(2*self.config.gram_emb_size+self.config.emb_size), config.n_classes, name='seq_fc')
        else:
            seq_fc_layer = layers.SeqFCLayer(len(region_s)*(2*self.config.gram_emb_size+self.config.emb_size), config.n_classes, name='seq_fc')

        seq_logits = seq_fc_layer(h)
        # seq_logits = tf.layers.dense(h, n_classes)

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
            if hasattr(self.config, 'pre_classification') and self.config.pre_classification:
                # logits_rep = rep
                rep = tf.math.sigmoid(rep)
                logits_rep = layers.SeqFCLayer(self.config.rep_size, config.n_classes, name='rep_seq_fc')(rep)
                self.loss_op += tf.contrib.seq2seq.sequence_loss(logits_rep, label, weights, average_across_timesteps=True, \
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
    pass

if '__main__' == __name__:
    main()

