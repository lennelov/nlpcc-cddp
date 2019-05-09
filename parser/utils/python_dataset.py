#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import random
import logging
import fileinput
import collections
import tensorflow as tf

from utils.tools import MapTable
from utils.tools import tf_type_mapping
from utils.tools import python_type_mapping
from utils.tools import Config

logger = logging.getLogger(__name__)


class PythonDataset(object):
    """PythonDataset"""

    def __init__(self, config):
        self._config = config
        self._map_tables = {}
        self._built = False

    def build(self, inputs=None):
        """build parse function and placeholders
        Return: function, dict
        """
        if self._built:
            logger.warn('Rebuilding dataset, ignored.')
            return

        # Load map tables if exsit
        self._load_map_tables()

        # Build dict of placeholds 
        self._build_inputs(inputs)

        self.prefetched = False
        self.prefetched_data = None

        self._built = True

    def _load_map_tables(self):
        if not hasattr(self._config, 'map_tables'):
            return

        for k, v in self._config.map_tables.as_dict().items():
            path = None
            key_column_index = tf.contrib.lookup.TextFileIndex.WHOLE_LINE
            value_column_index = tf.contrib.lookup.TextFileIndex.LINE_NUMBER
            delimiter = '\t'
            default_value = 1
            key_type = 'string'
            value_type = 'int32'

            if isinstance(v, str):
                items = v.strip().split(',')
                if len(items) == 2:
                    path, default_value = items
                    default_value = int(default_value)
                else:
                    path = items[0]

            else:
                if hasattr(v, 'path'):
                    path = v.path

                if hasattr(v, 'key_column_index'):
                    key_column_index = v.key_column_index

                if hasattr(v, 'value_column_index'):
                    value_column_index = v.value_column_index

                if hasattr(v, 'default_value'):
                    default_value = v.default_value

                if hasattr(v, 'delim'):
                    delimiter = v.delim

                if hasattr(v, 'value_type'):
                    value_type = v.value_type

                if hasattr(v, 'key_type'):
                    key_type = v.key_type

            self._map_tables[k] = MapTable(path,
                                           key_column_index=key_column_index,
                                           value_column_index=value_column_index,
                                           default_value=default_value,
                                           delimiter=delimiter,
                                           key_type=key_type,
                                           value_type=value_type)
            self._map_tables[k].load()
        return

    def _build_inputs(self, given_inputs=None):
        inputs = {}
        for slot_name, slot in self._config.slots.as_dict().items():
            shape, dtype = self._slot_to_placeholder_meta(slot)
            inputs[slot_name] = tf.placeholder(shape=[None] + shape, dtype=dtype, name=slot_name + '_placeholder')
            if not hasattr(slot, 'index'):
                logger.info('%s has not index specified, will be skip during pasing', slot_name)
                

        if given_inputs:
            inputs.update(given_inputs)

        logger.debug('Dataset Inputs: %s', inputs)
        self.inputs = inputs

    def _slot_to_placeholder_meta(self, slot):
        """slot_to_placeholder
        Args:
            slot (config):
        Returns:
            (list, tf.dtype) tuple: shape, dtype
        """
        assert hasattr(slot, 'type'), 'slot %s does not have a type.' % slot
        if slot.type == 'value':
            if hasattr(slot, 'value_type'):
                dtype, _ = tf_type_mapping(slot.value_type)
                tp, _ = python_type_mapping(slot.value_type)
                slot.value_type = tp
            if hasattr(slot, 'map_table'):
                dtype = self._map_tables[slot.map_table].value_dtype
                tp = self._map_tables[slot.map_table].value_type

            return [], dtype

        shape = []
        if slot.type == 'sequence':
            max_length = slot.max_length if hasattr(slot, 'max_length') else None

            assert hasattr(slot, 'value_type') or hasattr(slot, 'map_table'), slot

            if hasattr(slot, 'map_table'):
                assert slot.map_table in self._map_tables, "Unknown map table: %s" % slot.map_table
                dtype = self._map_tables[slot.map_table].value_dtype
            elif isinstance(slot.value_type, str):
                dtype, _ = tf_type_mapping(slot.value_type)
                tp, _ = python_type_mapping(slot.value_type)
                slot.value_type = tp
            else:
                shape, dtype = self._slot_to_placeholder_meta(slot.value_type)

            return [max_length] + shape, dtype

    def fetch_all(self):
        path = self._config.path
        if isinstance(path, str):
            path = [path]

        for p in path:
            assert os.path.exists(p), 'Data file[%s] not found' % p

        data = {}
        for buffer_data in self._fetched_buffer_iter(path, self._config.buffer_size):
            for k in buffer_data:
                if self.inputs[k] not in data:
                    data[self.inputs[k]] = []
                data[self.inputs[k]].extend(buffer_data[k])
        return data

    def make_iter(self, batch_size, fixed_batch_size=None):
        path = self._config.path
        if isinstance(path, str):
            path = [path]

        for p in path:
            assert os.path.exists(p), 'Data file[%s] not found' % p

        if fixed_batch_size is None:
            fixed_batch_size = self._config.fixed_batch_size if hasattr(self._config, 'fixed_batch_size') else False

        if not fixed_batch_size:
            margin = 1
        else:
            margin = batch_size

        if hasattr(self._config, 'prefetch') and self._config.prefetch:
            # Fetch when not fetched
            if not self.prefetched:
                self._prefetch_data(path)

            data_index = range(len(self.prefetched_data.values()[0]))
            if self._config.shuffle:
                logger.debug('Shuffling prefetched data, size: %s ...', len(data_index))
                start = time.time()
                random.shuffle(data_index)
                logger.debug('Shuffled, %s s elapsed', (time.time() - start))

            i = 0
            while i + margin <= len(data_index):
                data = {}
                indexs = data_index[i: i + batch_size]
                for k in self.prefetched_data:
                    data[self.inputs[k]] = [self.prefetched_data[k][index] for index in indexs]
                i += batch_size
                yield data
            self.prefetched = True

        else:
            for buffer_data in self._fetched_buffer_iter(path, self._config.buffer_size):
                data_index = range(len(buffer_data.values()[0]))
                if self._config.shuffle:
                    logger.debug('Shuffling buffer data, size: %s ...', len(data_index))
                    start = time.time()
                    random.shuffle(data_index)
                    logger.debug('Shuffled, %s s elapsed', (time.time() - start))

                i = 0
                while i + margin <= len(data_index):
                    data = {}
                    indexs = data_index[i: i + batch_size]
                    for k in buffer_data:
                        data[self.inputs[k]] = [buffer_data[k][index] for index in indexs]
                    i += batch_size
                    yield data
    
    def post_process(self, batch):
        return batch

    def _fetched_buffer_iter(self, path, buffer_size):
        logger.info('Start fetch dataset path:%s, max buffer size: %d', path, buffer_size)
        line_number = 0
        files = fileinput.FileInput(files=path)

        finished = False
        while not finished:
            fetched_data = {}
            for slot_name, slot in self._config.slots.as_dict().items():
                if not hasattr(slot, 'index'): 
                    continue

                fetched_data[slot_name] = []

            for i in xrange(buffer_size):
                try:
                    line = files.next()
                    if len(line) <= 1:
                        continue

                except Exception as e:
                    logger.debug('End of file, %s', e)
                    logger.debug('Total lines: %d', line_number)
                    finished = True
                    break

                if line_number and line_number % 100000 == 0:
                    logger.info("%d lines loaded ...", line_number + 1)

                data = self._parse(line)
                for k in fetched_data:
                    fetched_data[k].extend(data[k])

                line_number += 1

            if hasattr(self._config, 'negative_sampling'):
                logger.info('Start negative sampling ...')
                slot_key = self._config.negative_sampling.slot_key
                slots_keep = self._config.negative_sampling.slots_keep
                slots_copy = self._config.negative_sampling.slots_copy
                slots_assign = self._config.negative_sampling.slots_assign
                rate = self._config.negative_sampling.sampling_rate
                base_times = int(rate)
                prob = rate - int(rate)

                # Check covered all slots
                all_slots = {key: False for key in self._config.slots} 
                for slot_name in slots_keep:
                    all_slots[slot_name] = True

                for slot_name in slots_copy:
                    all_slots[slot_name] = True

                for slot_name in slots_assign:
                    all_slots[slot_name] = True

                assert all(all_slots.values()), 'negative sampling must cover all slots:%s' % all_slots
                
                sampled_data = {}
                for slot_name in self._config.slots:
                    sampled_data[slot_name] = []

                for i, key in enumerate(fetched_data[slot_key]):
                    if fetched_data['label'] == 0:
                        continue

                    times = base_times if random.uniform(0, 1) > prob else base_times + 1 
                    for time in xrange(times):
                        index = random.randint(0, len(fetched_data[slot_name]) - 1)
                        if fetched_data[slot_name][index] == key:
                            for _ in xrange(10):
                                index = random.randint(0, len(fetched_data[slot_name]))
                                if fetched_data[slot_name][index] != key:
                                    break
                       
                        for slot in slots_copy:
                            sampled_data[slot].append(fetched_data[slot][index])

                        for slot in slots_keep:
                            sampled_data[slot].append(fetched_data[slot][i])
                        
                        for slot in slots_assign: 
                            sampled_data[slot].append(slots_assign[slot])

                        for slot in fetched_data:
                            sampled_data[slot].append(fetched_data[slot][i])
                
                for slot_name in fetched_data:
                    fetched_data[slot_name].extend(sampled_data[slot_name])

                logger.info('%d lines negtive sampled.', len(sampled_data[slot_key]))

            fetched_size = -1
            for slot_name in fetched_data:
                if fetched_size < 0:
                    fetched_size = len(fetched_data[slot_name])
                assert len(fetched_data[slot_name]) == fetched_size, 'slot %s size %d is not consistant to %d' % (slot_name, len(fetched_data[slot_name]), fetched_size)

            fetched_data = self.post_process(fetched_data)
            yield fetched_data

    def _prefetch_data(self, path):
        logger.info("Start prefetch dataset, path:%s", path)
        self.prefetched_data = {}
        for slot_name in self._config.slots:
            self.prefetched_data[slot_name] = []

        start = time.time()
        files = fileinput.FileInput(files=path)
        for line_number, line in enumerate(files):
            if line_number and line_number % 100000 == 0:
                logger.info("%d_prefetch_data d lines loaded ...", line_number + 1)

            data = self._parse(line)
            for k in self.prefetched_data:
                self.prefetched_data[k].extend(data[k])

        logger.info("Total %d lines loaded, cost %.2f s.", line_number + 1, (time.time() - start))
        self.prefetched = True

    def parse_slot(self, slot_data, slot):
        expand_lines = 1
        if slot.type == 'sequence':
            assert hasattr(slot, 'value_type') or hasattr(slot, 'map_table'), slot
            if hasattr(slot, 'value_type') and isinstance(slot.value_type, Config):
                new_slot_data = []
                for ele in slot_data.split(slot.delim):
                    tmp_data, tmp_line = self.parse_slot(ele, slot.value_type)
                    new_slot_data.append(tmp_data[0])
                slot_data = new_slot_data

            else:
                if slot.delim:
                    items = slot_data.split(slot.delim)
                else:
                    items = [ch.encode('utf-8') for ch in slot_data.decode('utf-8')]

                items = filter(lambda x: x, items)
                
                if hasattr(slot, 'map_table'):
                    slot_data = map(lambda ele: self._map_tables[slot.map_table].lookup(ele), items)
                else:
                    slot_data = map(lambda ele: slot.value_type(ele), items)

            if hasattr(slot, 'sliding_window'):
                if hasattr(slot, 'pad'):
                    padding = [slot.pad] * (slot.sliding_window - len(slot_data))
                    slot_data = slot_data + padding

                expand_lines = len(slot_data) - slot.sliding_window + 1
                slot_data = [slot_data[i: i + slot.sliding_window] for i in xrange(expand_lines)]

            elif hasattr(slot, 'pad'):
                padding = [slot.pad] * (slot.max_length - len(slot_data))
                slot_data = slot_data + padding
                slot_data = [slot_data[:slot.max_length]]

            elif hasattr(slot, 'list_pad') and slot.list_pad == True:
                tmp_data, tmp_line = self.parse_slot('', slot.value_type)
                padding = [tmp_data[0]] * (slot.max_length - len(slot_data))
                slot_data = slot_data + padding
                slot_data = [slot_data[:slot.max_length]]

            else:
                slot_data = [slot_data]

        if slot.type == 'value':
            assert hasattr(slot, 'value_type') or hasattr(slot, 'map_table'), slot
            if hasattr(slot, 'value_type'):
                slot_data = [slot.value_type(slot_data)]
            if hasattr(slot, 'map_table'):
                slot_data = [self._map_tables[slot.map_table].lookup(slot_data)]

        return slot_data, expand_lines

    def _parse(self, line):
        items = line.split(self._config.delim)
        items[-1] = items[-1].rstrip()
        data = {}
        expand_lines_list = {}
        max_expand_lines = -1

        for slot_name, slot in self._config.slots.as_dict().items():
            if not hasattr(slot, 'index'): 
                continue
            if slot.index >= len(items):
                logger.error('slot.index exceed items')
                logger.error(str(slot_name) + '\n' + str(slot) + '\n' + '\n'.join(items))
                assert slot.index < len(items), str(slot_name) + '\n' + str(slot) + '\n' + '\n'.join(items)
            data[slot_name], expand_lines = self.parse_slot(items[slot.index], slot)
            expand_lines_list[slot_name] = expand_lines
            max_expand_lines = expand_lines if expand_lines > max_expand_lines else max_expand_lines

        for slot_name in data:
            data[slot_name] = data[slot_name] * (max_expand_lines / expand_lines_list[slot_name])

        return data


def main():
    import sys

    conf_file = sys.argv[1]
    conf = Config(conf_file, 'yaml')
    ds = PythonDataset(conf.estimator.infer_dataset)

    ds.build()
    print 'ds built.'

    batch_iter = ds.make_iter(1)
    print 'ds make iter .'
    import numpy as np
    for batch in batch_iter:
        print len(batch)
        for k in batch:
            print '%s\t%s' % (k, np.array(batch[k]).shape), batch[k][0]
        x = raw_input('Press any key to continue ...')
        # print len(batch.values()[0])


if __name__ == '__main__':
    main()
