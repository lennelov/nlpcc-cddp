#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# Get POS to id table
word = set()
pos = set()
data_dir = ['./train.conllu', './dev.conllu', './test.conllu']
for dataset in data_dir:
    with open(dataset, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                line = line.split('\t')
                if line[3] not in pos:
                    pos.add(line[3])
                word.add(line[1])
fout = open('pos2id.table', 'w')
pos = list(pos)
for idx, item in enumerate(pos):
    fout.write(item+'\n')
fout.close()

# Get word to id table
emb_dir = 'embedding/embedding.50'
word_table = './word2id.table'
emb_out = 'embedding/emb.50'
fout = open(word_table, 'bw')
fout1 = open(emb_out, 'w')
fout.write('<POS>\n<UNK>\n'.encode('utf-8'))
fout1.write(' '.join(['0.000000']*50)+'\n'+' '.join(['0.000000']*50)+'\n')
with open(emb_dir, encoding='utf-8') as fin:
    for idx, line in enumerate(fin):
        if idx > 0:
            line = line.strip().split(' ')
            if line[0] in word:
                fout.write((line[0]+'\n').encode('utf-8'))
                fout1.write(' '.join(line[1:])+'\n')
fout.close()
fout1.close()