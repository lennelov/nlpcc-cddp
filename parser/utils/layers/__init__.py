#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from utils.layers.layer import Layer

from utils.layers.merge_score_layer import MergeScoreNormalizeLayer

from utils.layers.functions_layer import CosineSimLayer
from utils.layers.functions_layer import WeightSumLayer
from utils.layers.functions_layer import XorLayer
from utils.layers.functions_layer import MaskMMLayer
from utils.layers.functions_layer import RowWiseTop_K_Average_PoolingLayer
from utils.layers.functions_layer import GumbelSoftmaxLayer
from utils.layers.functions_layer import masked_softmax

from utils.layers.embedding_layer import EmbeddingLayer
from utils.layers.embedding_layer import WindowPoolEmbeddingLayer
from utils.layers.embedding_layer import InitializedEmbeddingLayer
from utils.layers.embedding_layer import ScalarRegionEmbeddingLayer

from utils.layers.seq_cross_layer import SeqCrossLayer
from utils.layers.seq_cross_layer import SeqCosineLayer
from utils.layers.seq_cross_layer import SeqMatchLayer
from utils.layers.self_attention_layer import SelfAttentionLayer
from utils.layers.self_attention_layer import SelfImportanceLayer
from utils.layers.attention_layer import AttentionLayer
from utils.layers.attention_layer import SymAttentionLayer
from utils.layers.attention_layer import MultiHeadsDotProductAttentionLayer

from utils.layers.conv_layer import ConvLayer
from utils.layers.conv_layer import Conv2DLayer

from utils.layers.recurrent_layer import BiLSTMLayer
from utils.layers.recurrent_layer import BiGRULayer
from utils.layers.recurrent_layer import CompositeGRULayer
from utils.layers.recurrent_layer import SpatialGRULayer

from utils.layers.transformer_encoder import TransformerEncoderLayer
from utils.layers.transformer_encoder import TransformerDecoderLayer
from utils.layers.transformer_encoder import FeedForwardNetworks
from utils.layers.attention_layer import DotProductAttentionLayer

from utils.layers.pool_layer import DynamicPool2DLayer
from utils.layers.cosin_layer import CosinLayer
from utils.layers.concat_layer import ConcatLayer
from utils.layers.vsum_layer import VSumLayer
from utils.layers.vsum_layer import WeightedVSumLayer

from utils.layers.region_alignment_layer import RegionAlignmentLayer
from utils.layers.region_alignment_layer import WindowAlignmentLayer

from utils.layers.fc_layer import FCLayer
from utils.layers.fc_layer import SeqFCLayer

from utils.layers.mask_layer import SeqMaskLayer

from utils.layers.loss_layer import ReducedPairwiseHingeLossLayer
from utils.layers.loss_layer import CrossEntropyLossLayer
from utils.layers.loss_layer import WeightedCrossEntropyLossLayer

from utils.layers.metric_layer import DefaultClassificationMetricLayer
from utils.layers.metric_layer import MultiClassificationMetricLayer
from utils.layers.metric_layer import DefaultLossMetricLayer
from utils.layers.metric_layer import UASMetricLayer

from utils.layers.pointer_network_layer import PointerNetDecoder
