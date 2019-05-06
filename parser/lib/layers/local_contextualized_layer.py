#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from lib.layers.layer import Layer
from lib.layers.attention_layer import AttentionLayer, MultiHeadsDotProductAttentionLayer
from lib.layers.region_alignment_layer import RegionAlignmentLayer
from lib.layers.fc_layer import FCLayer

class LocalContextualizedLayer(Layer):
    def __init__(self, vocab_size, region_size, emb_size,
                aggregate_method_within_region='attention', n_heads=None, 
                name="LocalContextualizedLayer", shared_name=None,
                have_context_unit=True,
                initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32), 
                trainable=True, **kwargs):
        super(LocalContextualizedLayer, self).__init__(name, **kwargs)
        if have_context_unit:
            if shared_name is None:
                with tf.variable_scope(name):
                    self.U = tf.get_variable('U', shape=[vocab_size, region_size, emb_size],\
                        dtype=tf.float32, initializer=initializer, trainable=trainable)
            else:
                with tf.variable_scope(shared_name, reuse=tf.AUTO_REUSE):
                    self.U = tf.get_variable('U', shape=[vocab_size, region_size, emb_size],\
                        dtype=tf.float32, initializer=initializer, trainable=trainable)  
        
        with tf.variable_scope(name):
            if aggregate_method_within_region == "attention":
                self.attention_layer = AttentionLayer(emb_size, emb_size, name="attention_layer", trainable=trainable)
            if aggregate_method_within_region == 'multihead_attention':
                assert n_heads is not None
                self.n_heads = n_heads
                self.attention_layer = MultiHeadsDotProductAttentionLayer(hidden_size=emb_size, 
                    d_k=int(emb_size/n_heads), 
                    d_v=int(emb_size/n_heads),
                    n_heads=n_heads,
                    trainable=trainable
                )
        self.region_size = region_size
        self.emb_size = emb_size
        self.aggregate_method_within_region = aggregate_method_within_region
        self.have_context_unit = have_context_unit

    def _forward(self, seq, previous_h, aligned_idx, x_mask=None, y_mask=None):
        """
        previous_h : intermediate representation from previous layer, [batch_size, seq_len, emb_size]
        aligned_idx: used for getting intermediate representation in the window of each word, [batch_size, seq_len, region_size]
        x_mask: [batch_size*seq_len, 1, region_size]
        y_mask: [batch_size*seq_len, region_size] both used for multihead-attention
        """
        # add zeros pad before representation of each sequence, [batch_size, seq_len, emb_size] -> [batch_size, seq_len+1, emb_size]
        batch_size = tf.shape(previous_h)[0]
        seq_len = tf.shape(previous_h)[1]
        zeros_pad = tf.zeros(shape=[batch_size, 1, self.emb_size], dtype=previous_h.dtype)
        previous_h = tf.concat([zeros_pad, previous_h], axis=1)
        # tile [batch_size, seq_len+1, emb_size] in axis 1 to convert it to [batch_size, seq_len, seq_len+1, emb_size]
        previous_h = tf.tile(tf.expand_dims(previous_h, 1), [1, seq_len, 1, 1])
        # [batch_size, seq_len, region_size, emb_size]
        context = tf.batch_gather(previous_h, aligned_idx)
        if self.have_context_unit:
            context_unit = tf.nn.embedding_lookup(self.U, seq)
            context = context * context_unit

        if self.aggregate_method_within_region == "reduce_mean":
            h = tf.reduce_mean(context, axis=2)
        elif self.aggregate_method_within_region == "reduce_max":
            h = tf.reduce_max(context, axis=2)
        elif self.aggregate_method_within_region == "attention":
            seq_len = tf.shape(context)[1]
            context = tf.reshape(context, shape=[-1, self.region_size, self.emb_size])
            attented_key = context[:, int(self.region_size/2), :]
            _, attented_context = self.attention_layer(state=attented_key, seq=context) # [batch_size*seq_len, region_size, emb_size]
            attented_context = tf.reshape(attented_context, shape=[-1, seq_len, self.region_size, self.emb_size])
            h = tf.reduce_sum(attented_context, axis=2)
        elif self.aggregate_method_within_region == "multihead_attention":
            batch_size = tf.shape(context)[0]
            seq_len = tf.shape(context)[1]
            context = tf.reshape(context, shape=[batch_size*seq_len, self.region_size, self.emb_size])
            attented_key = tf.expand_dims(context[:, int(self.region_size/2), :], 1) # [batch_size*seq_len, 1, emb_size]
            _, attented_context = self.attention_layer(
                x=attented_key,
                y=context,
                x_mask=x_mask,
                y_mask=y_mask
            )
            print "attented_context: ", attented_context
            h = tf.reshape(attented_context, [-1, seq_len, self.emb_size])
        
        return h
            
class LocalContextualizedEncoder(Layer):
    def __init__(self, vocab_size, region_size, emb_size, num_layers,
            share_context_unit=False,
            aggregate_method_within_region='attention', 
            n_heads=None,
            fc_between_layer = True,
            name="LocalContextualizedEncoder",
            initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32), 
            only_first_layer_context_unit=False,
            trainable=True,
            **kwargs):
        super(LocalContextualizedEncoder, self).__init__(name, **kwargs)
        self.region_alignment_layer = RegionAlignmentLayer(region_size)
        self.layer_list = []
        self.fc_layer_list = []
        for layer_i in range(num_layers):
            shared_name = name if share_context_unit else None
            if only_first_layer_context_unit and layer_i != 0:
                have_context_unit = False
            else:
                have_context_unit = True
            self.layer_list.append(LocalContextualizedLayer(
                vocab_size,
                region_size, 
                emb_size,
                aggregate_method_within_region=aggregate_method_within_region,
                n_heads=n_heads,
                name="LocalContextualizedLayer{}".format(layer_i),
                shared_name=shared_name,
                have_context_unit=have_context_unit,
                trainable=trainable
            ))
            if fc_between_layer:
                self.fc_layer_list.append(FCLayer(emb_size, emb_size, name="fc_L{}".format(layer_i), trainable=trainable))

        self.region_size = region_size
        self.num_layers = num_layers
        self.fc_between_layer = fc_between_layer
        self.aggregate_method_within_region = aggregate_method_within_region

    def _forward(self, seq, h):
        """
        seq: [batch_size, seq_len], the input sequence
        h: [batch_size, seq_len, emb_size] word embeddings of the input sequence
        """
        batch_size = tf.shape(seq)[0]
        seq_len = tf.shape(seq)[1]
        emb_size = tf.shape(h)[2]
        range_x = tf.reshape(tf.tile(tf.expand_dims(tf.range(1, seq_len+1), 0), [batch_size, 1]), tf.shape(seq))
        # in current version region_alignment_layer, pad position will be filled with 0
        # [batch_size, seq_len, region_size]
        aligned_x = tf.cast(self.region_alignment_layer(range_x), dtype=tf.int32)
        
        x_mask, y_mask = None, None
        if self.aggregate_method_within_region == "multihead_attention":
            tmp_aligned_x = tf.reshape(aligned_x, [batch_size*seq_len, self.region_size]) # [batch_size*seq_len, region_size]
            condition = tf.math.not_equal(tmp_aligned_x, 0) # see 0 as pad, mask the padded position
            one_tensors = tf.ones(shape=tf.shape(tmp_aligned_x), dtype=h.dtype)
            zero_tensors = tf.zeros(shape=tf.shape(tmp_aligned_x), dtype=h.dtype)
            y_mask = tf.where(condition, one_tensors, zero_tensors)
            x_mask = tf.expand_dims(y_mask[:, int(self.region_size/2)], 1)

        for layer_i in range(self.num_layers):
            h = self.layer_list[layer_i](
                seq=seq,
                previous_h=h,
                aligned_idx=aligned_x,
                x_mask=x_mask,
                y_mask=y_mask
            )
            if layer_i < self.num_layers-1:
                fc_layer = self.fc_layer_list[layer_i]
                seq_len = tf.shape(h)[1]
                emb_size = tf.shape(h)[2]
                h = tf.reshape(fc_layer(tf.reshape(h,[-1, emb_size])), [batch_size, seq_len, emb_size])
                h = tf.tanh(h)
        return h