#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def get_data(config):
    word2id = get_table(config.word2id)
    pos2id = get_table(config.pos2id)
    
    idx = []
    word = []
    upos = []
    xpos = []
    head = []
    length = []
    with open(config['data_dir'], encoding='utf-8') as fin:
        idx_sent = []
        word_sent = []
        upos_sent = []
        xpos_sent = []
        head_sent = []
        for line in fin:
            line = line.strip()
            if line != '':
                line = line.split('\t')
                idx_sent.append(int(line[0]))
                word_sent.append(word2id[line[1]])
                upos_sent.append(pos2id[line[3]])
                xpos_sent.append(pos2id[line[4]])
                head_sent.append(int(line[6]))
            else:
                len_sent = len(idx)
                if len_sent > 0:
                    idx.append(idx_sent)
                    word.append(word_sent)
                    upos.append(upos_sent)
                    xpos.append(xpos_sent)
                    head.append(head_sent)
                    length.append(len_sent)
                    
                    idx_sent = []
                    word_sent = []
                    upos_sent = []
                    xpos_sent = []
                    head_sent = []
    dataset = {'idx':idx,
               'word':word,
               'upos':upos,
               'xpos':xpos,
               'head':head,
               'length':length
               }
    
    return dataset
    
def get_table(table_dir):
    table = {}
    with open(table_dir, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip.split('\t')
            table[line[1]] = int(line[0])

    return table

