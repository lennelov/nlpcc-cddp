#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from utils.layers.layer import Layer

from utils.layers.embedding_layer import EmbeddingLayer
from utils.layers.embedding_layer import InitializedEmbeddingLayer

from utils.layers.fc_layer import FCLayer
from utils.layers.fc_layer import SeqFCLayer

from utils.layers.metric_layer import DefaultClassificationMetricLayer
from utils.layers.metric_layer import MultiClassificationMetricLayer
from utils.layers.metric_layer import DefaultLossMetricLayer
from utils.layers.metric_layer import UASMetricLayer