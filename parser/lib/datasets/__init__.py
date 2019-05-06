#!/usr/bin/env python
#-*- coding: utf-8 -*-

from python_dataset import PythonDataset

def get_dataset(config):
    if config.type == 'PythonDataset':
        return PythonDataset(config)

    raise KeyError('unknown dataset type: %s' % config.type)
