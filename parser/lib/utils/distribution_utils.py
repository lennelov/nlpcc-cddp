#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import json
import logging
import tensorflow as tf
# import tensorflow.distribute
# import tensorflow.contrib.distribute

logger = logging.getLogger(__name__)


def set_arnold_tf_config():
    """Set TF_CONFIG accrodding to arnold's environment variables"""
    ps_hosts = os.environ['ARNOLD_SERVER_HOSTS'].split(",")
    worker_hosts = os.environ['ARNOLD_WORKER_HOSTS'].split(",")
    role = os.environ['ARNOLD_ROLE']
    tf_config = json.dumps(
        {"cluster": {"worker": worker_hosts, "ps": ps_hosts},
         "task": {"type": role, "index": int(os.environ['ARNOLD_ID'])}})

    os.environ['TF_CONFIG'] = tf_config
    logger.info("Set environment variable: TF_CONFIG: %s", tf_config)


def get_distribution_strategy(conf):
    """Return a DistributionStrategy for running the model.
    Args:

    conf:
      distribution_strategy: a string specify which distribution strategy to use.
        Accepted values are 'off', 'default', 'one_device', 'mirrored',
        'parameter_server', 'multi_worker_mirrored', case insensitive. 'off' means
        not to use Distribution Strategy; 'default' means to choose from
        `MirroredStrategy`, `MultiWorkerMirroredStrategy`, or `OneDeviceStrategy`
        according to the number of GPUs and number of workers.
      num_gpus: Number of GPUs to run this model.
      num_workers: Number of workers to run this model.
      all_reduce_alg: Optional. Specify which algorithm to use when performing
        all-reduce. See tf.contrib.distribute.AllReduceCrossDeviceOps for
        available algorithms when used with `mirrored`, and
        tf.distribute.experimental.CollectiveCommunication when used with
        `multi_worker_mirrored`. If None, DistributionStrategy will choose based
        on device topology.
    Returns:
      tf.distribute.DistibutionStrategy object.
    Raises:
      ValueError: if `distribution_strategy` is 'off' or 'one_device' and
        `num_gpus` is larger than 1; or `num_gpus` is negative.
    """
    distribution_strategy = conf.strategy 
    num_gpus = conf.num_gpus if hasattr (conf, 'num_gpus') else 0 
    num_workers = conf.num_workers if hasattr (conf, 'num_workers') else 1
    all_reduce_alg = conf.all_reduce_alg if hasattr (conf, 'all_reduce_alg') else None
                
    if num_gpus < 0:
        raise ValueError("`num_gpus` can not be negative.")

    distribution_strategy = distribution_strategy.lower()

    if distribution_strategy == "off":
        if num_gpus > 1 or num_workers > 1:
            raise ValueError(
                "When {} GPUs and  {} workers are specified, distribution_strategy "
                "flag cannot be set to 'off'.".format(num_gpus, num_workers))
        return None

    if distribution_strategy == "multi_worker_mirrored" and num_workers > 1:
        # # no this attribute in tf 1.13.1
        # return tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)
        return tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker = num_gpus)
    
    if distribution_strategy == "collective_all_reduce":
        return tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=num_gpus)

    if (distribution_strategy == "one_device" or
            (distribution_strategy == "default" and num_gpus <= 1)):
        if num_gpus == 0:
            return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")

        if num_gpus > 1:
            raise ValueError("`OneDeviceStrategy` can not be used for more than "
                             "one device.")

        return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")

    if distribution_strategy in ("mirrored", "default"):
        if num_gpus == 0:
            assert distribution_strategy == "mirrored"
            devices = ["device:CPU:0"]
        else:
            devices = ["device:GPU:%d" % i for i in range(num_gpus)]

        return tf.distribute.MirroredStrategy(devices=devices)

    if distribution_strategy == "parameter_server":
        # return tf.distribute.experimental.ParameterServerStrategy()
        return tf.contrib.distribute.ParameterServerStrategy(num_gpus_per_worker=num_gpus)

    raise ValueError(
        "Unrecognized Distribution Strategy: %r" % distribution_strategy)


def main():
    pass


if __name__ == '__main__':
    main()
