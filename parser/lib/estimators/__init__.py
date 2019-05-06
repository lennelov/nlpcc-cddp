#!/usr/bin/env python
#-*- coding: utf-8 -*-

import logging

from python_estimator import PythonEstimator
from python_seq_estimator import PythonSequenceLabellingEstimator


def get_estimator(config, model):
    logging.debug('estimator type: %s', config.type) 

    if config.type == 'PythonEstimator':
        return PythonEstimator(config, model)

    if config.type == 'PythonSequenceLabellingEstimator':
        return PythonSequenceLabellingEstimator(config, model)

    raise KeyError('unknown dataset type:%s' % config.type)