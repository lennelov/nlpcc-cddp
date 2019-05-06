#!/usr/bin/env python
#-*- coding: utf-8 -*-


import json
import yaml

class Config(object):
    """Config"""
    def __init__(self, config_object, config_type='dict'):
        assert config_type in ['dict', 'json', 'yaml']

        if config_type == 'dict':
            config = config_object

        if config_type == 'json':
            with open(config_object, 'r') as fin:
                config = json.load(fin)

        if config_type == 'yaml':
            yaml.Loader.ignore_aliases = lambda *args : True
            with open(config_object, 'r') as fin:
                config = yaml.load(fin)
        
        #print 'yaml', config
        if config:
            self._dict = config
            self._update(config)

    def as_dict(self):
        return self._dict

    def add(self, key, value):
        """add

        Args:
                key(type):
                value(type):
        Returns:
                type:
        """
        self._dict[key] = value
        self.__dict__[key] = value
        
    def _update(self, config):
        if not isinstance(config, dict):
            return

        for key in config:
            if isinstance(config[key], dict):
                config[key] = Config(config[key].copy())

            if isinstance(config[key], list):
                config[key] = [Config(x.copy()) if isinstance(x, dict) else x for x in config[key]]

        self._dict.update(config)
        self.__dict__.update(config.items())

    def __iter__(self):
        return self._dict.__iter__()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
       return json.dumps(yaml.load('%s' % self._dict), indent=4)

def main():
    """unit test for main"""
    import sys
    conf = Config(sys.argv[1], 'yaml')
    print conf

if __name__ == '__main__':
    main()
