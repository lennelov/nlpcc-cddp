#!/usr/bin/env python
#-*- coding: utf-8 -*-

import logging

from python_estimator import PythonEstimator

def get_estimator(config, model):
    logging.debug('estimator type: %s', config.type) 
    return PythonEstimator(config, model)