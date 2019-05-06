#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from lib.layers.layer import Layer

from lib.layers.merge_score_layer import MergeScoreNormalizeLayer

from lib.layers.functions_layer import CosineSimLayer
from lib.layers.functions_layer import WeightSumLayer
from lib.layers.functions_layer import XorLayer
from lib.layers.functions_layer import MaskMMLayer
from lib.layers.functions_layer import RowWiseTop_K_Average_PoolingLayer
from lib.layers.functions_layer import GumbelSoftmaxLayer
from lib.layers.functions_layer import masked_softmax


from lib.layers.embedding_layer import EmbeddingLayer
from lib.layers.embedding_layer import WindowPoolEmbeddingLayer
from lib.layers.embedding_layer import InitializedEmbeddingLayer
from lib.layers.embedding_layer import ScalarRegionEmbeddingLayer
from lib.layers.embedding_layer import MultiRegionEmbeddingLayer
from lib.layers.embedding_layer import WordContextRegionEmbeddingLayer
from lib.layers.embedding_layer import ContextWordRegionEmbeddingLayer


from lib.layers.seq_cross_layer import SeqCrossLayer
from lib.layers.seq_cross_layer import SeqCosineLayer
from lib.layers.seq_cross_layer import SeqMatchLayer
from lib.layers.self_attention_layer import SelfAttentionLayer
from lib.layers.self_attention_layer import SelfImportanceLayer
from lib.layers.attention_layer import AttentionLayer
from lib.layers.attention_layer import SymAttentionLayer
from lib.layers.attention_layer import MultiHeadsDotProductAttentionLayer

from lib.layers.conv_layer import ConvLayer
from lib.layers.conv_layer import Conv2DLayer


from lib.layers.recurrent_layer import BiLSTMLayer
from lib.layers.recurrent_layer import BiGRULayer
from lib.layers.recurrent_layer import CompositeGRULayer
from lib.layers.recurrent_layer import SpatialGRULayer

from lib.layers.transformer_encoder import TransformerEncoderLayer
from lib.layers.transformer_encoder import TransformerDecoderLayer
from lib.layers.transformer_encoder import FeedForwardNetworks
from lib.layers.attention_layer import DotProductAttentionLayer


from lib.layers.pool_layer import DynamicPool2DLayer
from lib.layers.cosin_layer import CosinLayer
from lib.layers.concat_layer import ConcatLayer
from lib.layers.vsum_layer import VSumLayer
from lib.layers.vsum_layer import WeightedVSumLayer


from lib.layers.region_alignment_layer import RegionAlignmentLayer
from lib.layers.region_alignment_layer import WindowAlignmentLayer

from lib.layers.region_emb_layer import RegionEmbeddingLayer
from lib.layers.region_emb_layer import MultiRegionEmbeddingLayer

from lib.layers.fc_layer import FCLayer
from lib.layers.fc_layer import SeqFCLayer

from lib.layers.mask_layer import SeqMaskLayer

from lib.layers.loss_layer import ReducedPairwiseHingeLossLayer
from lib.layers.loss_layer import CrossEntropyLossLayer
from lib.layers.loss_layer import WeightedCrossEntropyLossLayer

from lib.layers.metric_layer import DefaultClassificationMetricLayer
from lib.layers.metric_layer import DefaultRankingMetricLayer
from lib.layers.metric_layer import MultiClassificationMetricLayer
from lib.layers.metric_layer import DefaultLossMetricLayer
from lib.layers.metric_layer import NERMetricLayer

from lib.layers.pointer_network_layer import PointerNetDecoder

from lib.layers.global_contextualized_layer import GlobalContextualizedLayer, GlobalContextualizedEncoder
from lib.layers.local_contextualized_layer import LocalContextualizedLayer, LocalContextualizedEncoder
