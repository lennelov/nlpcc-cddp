#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from .layer import Layer
from .embedding_layer import EmbeddingLayer
from .region_alignment_layer import WindowAlignmentLayer


class RegionEmbeddingLayer(Layer):
    """RegionEmbeddingLayer"""

    def __init__(self, vocab_size, emb_size, region_size,
                 initializer=None, name='region_emb_layer', **kwargs):
        super(RegionEmbeddingLayer, self).__init__(name=name, **kwargs)

        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()

        self._region_size = region_size
        self._U = self.get_variable(name + '_U',
                                    shape=[vocab_size, region_size, emb_size], dtype=tf.float32)
        self._region_merge_fn = tf.reduce_max

    def _forward(self, seq, seq_emb):

        region_radius = self._region_size / 2
        paddings = tf.constant(
            [[0, 0], [region_radius, region_radius], [0, 0]])
        paded_seq_emb = tf.pad(seq_emb, paddings, "CONSTANT")
        region_aligned_emb = WindowAlignmentLayer(
            self._region_size)(paded_seq_emb)
        context_units = tf.nn.embedding_lookup(self._U, seq)

        projected_emb = region_aligned_emb * context_units
        return self._region_merge_fn(projected_emb, axis=2)


class MultiRegionEmbeddingLayer(Layer):
    """MultiRegionEmbeddingLayer"""

    def __init__(self, vocab_size, emb_size, region_sizes,
                 initializer=None, share=True, name='multi_region_emb_layer', **kwargs):

        super(MultiRegionEmbeddingLayer, self).__init__(name=name, **kwargs)

        if not initializer:
            initializer = tf.contrib.layers.variance_scaling_initializer()

        # 7, 5, 3
        self._region_sizes = sorted(region_sizes, reverse=True)
        self._share = share

        if self._share:
            self._U = self.get_variable(name + '_U',
                                        shape=[vocab_size, self._region_sizes[0], emb_size], dtype=tf.float32)
        else:
            self._U = {}
            for region_size in self._region_sizes:
                self._U[region_size] = self.get_variable(name + '_U_%d' % region_size,
                                                         shape=[vocab_size, region_size, emb_size], dtype=tf.float32)

        self._region_merge_fn = tf.reduce_max

    def _forward(self, seq, seq_emb):
        if self._share:
            projected_emb = \
                self._forward_with_region_size(
                    seq, seq_emb, self._region_sizes[0], self._U)
            region_embs = []
            for i in xrange(len(self._region_sizes)):
                margin = (self._region_sizes[0] - self._region_sizes[i]) / 2
                if margin:
                    p_e = projected_emb[:, :, margin:-margin]
                else:
                    p_e = projected_emb
                region_embs.append(self._region_merge_fn(p_e, axis=2))
            return tf.concat(region_embs, axis=2)

        region_embs = []
        for region_size in self._region_sizes:
            U = self._U[region_size]
            projected_emb = self._forward_with_region_size(
                seq, seq_emb, region_size, U)
            region_embs.append(self._region_merge_fn(projected_emb, axis=2))
        return tf.concat(region_embs, axis=2)

    def _forward_with_region_size(self, seq, seq_emb, region_size, U):
        region_radius = region_size / 2
        paddings = tf.constant(
            [[0, 0], [region_radius, region_radius], [0, 0]])
        paded_seq_emb = tf.pad(seq_emb, paddings, "CONSTANT")
        region_aligned_emb = WindowAlignmentLayer(region_size)(paded_seq_emb)
        context_units = tf.nn.embedding_lookup(U, seq)
        projected_emb = region_aligned_emb * context_units
        return projected_emb


def main():
    pass


if '__main__' == __name__:
    main()
    main()
