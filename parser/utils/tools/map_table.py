#!/usr/bin/env python
#-*- coding: utf-8 -*-


import logging
import tensorflow as tf
logger = logging.getLogger(__name__)

from utils.tools.funcs import tf_type_mapping
from utils.tools.funcs import python_type_mapping

class MapTable(object):
    def __init__(self, path, default_value="",
            default_values = None,
            key_column_index=0,
            value_column_index=1,
            delimiter='\t',
            key_type='str',
            value_type='int'):
        self.path = path
        #self.default_value = default_value
        
        if default_values is None:
            self.default_values = [default_value]
        
        self.cached_default_values = {}
        self.used_default_value_index = 0

        self.key_column_index = key_column_index
        self.value_column_index = value_column_index
        self.delimiter = delimiter

        self.key_dtype, _ = tf_type_mapping(key_type)
        self.value_dtype, _ = tf_type_mapping(value_type)

        self.key_type, _ = python_type_mapping(key_type)
        self.value_type, _ = python_type_mapping(value_type)
    
    def load(self):
        logger.debug('Start loading map table, path:%s, key: %d, value: %d, default_values:%s,'\
                'key_type:%s, value_type:%s',
                self.path, self.key_column_index, self.value_column_index, self.default_values,
                self.key_type, self.value_type)
        self.map_table = {}
        with open(self.path, 'r') as fin:
            for line_number, line in enumerate(fin):
                line = line.strip()
                items = line.split(self.delimiter)
                try:
                    if self.key_column_index == tf.contrib.lookup.TextFileIndex.WHOLE_LINE:
                        key = self.key_type(line)
                    
                    if self.key_column_index == tf.contrib.lookup.TextFileIndex.LINE_NUMBER:
                        key = line_number 
                    
                    if self.key_column_index > -1:
                        key = self.key_type(items[self.key_column_index])

                    if self.value_column_index == tf.contrib.lookup.TextFileIndex.WHOLE_LINE:
                        value = self.value_type(line)
                    
                    if self.value_column_index == tf.contrib.lookup.TextFileIndex.LINE_NUMBER:
                        value = line_number 
                    
                    if self.value_column_index > -1:
                        value = self.value_type(items[self.value_column_index])

                    self.map_table[key] = value
                except Exception as e:
                    logger.error('Load line[%d]: %s fail. %s', line_number, line, e)
                    raise e

        logger.debug('%s Lines loaded.', len(self.map_table))

    def check_lookup(self, key):
        return key in self.map_table

    def lookup(self, key):
        if key not in self.map_table:
            if key in self.cached_default_values:
                return self.cached_default_values[key]
            self.used_default_value_index = (self.used_default_value_index + 1) % len(self.default_values) 
            value = self.default_values[self.used_default_value_index]
            self.cached_default_values[key] = value
            return value
        return self.map_table[key]

    def reset_cache(self):  
        #self.used_default_value_index = 0
        self.cached_default_values = {}

def main():
    pass

if __name__ == '__main__':
    main()
