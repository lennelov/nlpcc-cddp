#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from .layer import Layer
from .region_alignment_layer import RegionAlignmentLayer


class EmbeddingLayer(Layer):
    """EmbeddingLayer"""

    def __init__(self, vocab_size, emb_size, trainable=True, name="embedding",
                 initializer=None, **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()

        self._W = self.get_variable('W', shape=[vocab_size, emb_size],
                                    initializer=initializer, trainable=trainable)

    def _forward(self, seq, zero_foward=False):
        if zero_foward:
            seq_mask = tf.cast(tf.stack([tf.sign(seq)] * self._emb_size, axis=-1), tf.float32)
            emb = tf.nn.embedding_lookup(self._W, seq)
            emb = emb * seq_mask
            return emb
        else:
            return tf.nn.embedding_lookup(self._W, seq)


class InitializedEmbeddingLayer(Layer):
    def __init__(self, vocab_size, emb_size, init_dict, trainable=False, name="embedding",
                 initializer=None, **kwargs):

        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size

        embedding = np.zeros([vocab_size, emb_size])
        with open(init_dict) as fin:
            for i, line in enumerate(fin):
                line_list = line.strip().split('\t')
                if len(line_list) == 1:
                    id, vec = i, [float(_) for _ in line_list[0].split()]
                else:
                    id, vec = int(line_list[0]), [float(_) for _ in line_list[1].split()]
                if len(vec) != emb_size or id >= vocab_size:
                    print
                    'Load pretrained emb: id:%s, len_vec:%s, line:%s', (id, len(vec), line)
                    assert False
                else:
                    embedding[id] = vec

        if trainable:
            self._W = self.get_variable('W', shape=[vocab_size, emb_size],
                                        initializer=tf.constant_initializer(embedding), trainable=trainable)
        else:
            self._W = tf.constant(embedding, dtype=tf.float32)

    def _forward(self, seq, zero_foward=False):
        if zero_foward:
            W = tf.concat((tf.zeros(shape=[1, self._emb_size]), self._W[1:, :]), 0)
            return tf.nn.embedding_lookup(W, seq)
        else:
            return tf.nn.embedding_lookup(self._W, seq)


class WindowPoolEmbeddingLayer(EmbeddingLayer):
    """WindowPoolEmbeddingLayer"""

    def __init__(self, vocab_size, emb_size, region_size,
                 region_merge_fn=None,
                 name="win_pool_embedding",
                 initializer=None,
                 **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable('K', shape=[vocab_size, region_size, emb_size],
                                    initializer=initializer)
        super(WindowPoolEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                                                       initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(
            WindowPoolEmbeddingLayer,
            self)._forward(region_aligned_seq)

        return self._region_merge_fn(region_aligned_emb, axis=2)


class ScalarRegionEmbeddingLayer(EmbeddingLayer):
    """WordContextRegionEmbeddingLayer(Scalar)"""

    def __init__(self, vocab_size, emb_size, region_size,
                 region_merge_fn=None,
                 name="scalar_region_embedding",
                 initializer=None,
                 **kwargs):
        Layer.__init__(self, name, **kwargs)
        self._emb_size = emb_size
        self._region_size = region_size
        self._region_merge_fn = region_merge_fn
        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        self._K = self.get_variable('K', shape=[vocab_size, region_size, 1],
                                    initializer=initializer)
        super(ScalarRegionEmbeddingLayer, self).__init__(vocab_size, emb_size, name,
                                                         initializer, **kwargs)

    def _forward(self, seq):
        # Region alignment embedding
        region_aligned_seq = RegionAlignmentLayer(self._region_size)(seq)
        region_aligned_emb = super(
            ScalarRegionEmbeddingLayer,
            self)._forward(region_aligned_seq)

        region_radius = self._region_size / 2
        trimed_seq = seq[:, region_radius: seq.get_shape()[1] - region_radius]
        context_unit = tf.nn.embedding_lookup(self._K, trimed_seq)

        projected_emb = region_aligned_emb * context_unit
        return self._region_merge_fn(projected_emb, axis=2)


def main():
    """main"""
    pass


if '__main__' == __name__:
    main()
