#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import logging
import warnings
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 默认为0：输出所有log信息
# 设置为1：进一步屏蔽INFO信息
# 设置为2：进一步屏蔽WARNING信息
# 设置为3：进一步屏蔽ERROR信息

import tensorflow as tf
import lib
from lib.utils import config
from lib.estimators import get_estimator
import models


# Supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('conf', help='conf file')
parser.add_argument('-action', default='train', help='action')

def evaluate(est):
    est.eval()

def infer(est):
    est.infer()

def main():
    """main"""

    warnings.simplefilter("ignore", DeprecationWarning)
    
    logger = logging.getLogger(__name__)
    args = parser.parse_args()
    conf = config.Config(os.path.join(args.conf), 'yaml')
    
    if hasattr(conf, 'logging'):
        log_dir = os.path.dirname(conf.logging.file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        lib.initialize_logging(open(conf.logging.file, 'w'), conf.logging.level)
    else:
        lib.initialize_logging(sys.stdout, 'DEBUG')

    if not conf.estimator.checkpoint_dir:
        conf.estimator.checkpoint_dir = 'local_results/' + args.conf

    logger.debug('Run with config: %s', args.conf)

    # Create Model
    model_map = {
        'DependencyModel': models.DependencyModel,
    }

    assert conf.model.model_name in model_map, 'unknown model name: %s' % conf.model.model_name
    model = model_map[conf.model.model_name](conf.model)

    # Create Estimator
    est = get_estimator(conf.estimator, model)

    # Execute actions
    # if args.action == 'export':
    #     est.export_model()

    if args.action == 'train':
        est.train()

    if args.action == 'eval':
        evaluate(est)

    if args.action == 'infer':
        infer(est)


if '__main__' == __name__:
    main()
