#!/usr/bin/env python
#-*- coding: utf-8 -*-


import tensorflow as tf

def tf_type_mapping(value_type_str, default_value=None):
    """tf_type_mapping
    Args:
        value_type_str (type):
        default_value (type):
    Returns:
        tuple (type, value): default_type, default_value
    """
    assert value_type_str in ['str', 'string', 'int', 'int32', 'int64', 'float', 'float32', 'float64'], value_type_str
    
    if value_type_str == "str" or value_type_str == "string":
        value_type = tf.string
        _default_value = ""

    if value_type_str == "int" or value_type_str == "int32":
        value_type = tf.int32
        _default_value = 0

    if value_type_str == "int64":
        value_type = tf.int64
        _default_value = 0

    if value_type_str == "float" or value_type_str == "float32":
        value_type = tf.float32
        _default_value = 0.0

    if value_type_str == "float64":
        value_type = tf.float64
        _default_value = 0.0
    
    if not default_value:
        default_value = _default_value

    return value_type, default_value


def python_type_mapping(value_type_str, default_value=None):
    """python_type_mapping
    Args:
        value_type_str (string):
        default_value (type):
    Returns:
        tuple (type, value): default_type, default_value
    """
    assert value_type_str in ['str', 'string', 'int', 'int32', 'int64', 'float', 'float32', 'float64'], value_type_str

    if value_type_str == "str" or value_type_str == "string":
        value_type = str
        default_value = ""

    if 'int' in value_type_str:
        value_type = int
        default_value = 0

    if 'float' in value_type_str:
        value_type = float
        default_value = 0.0

    return value_type, default_value


def main():
    pass

if __name__ == '__main__':
    main()
