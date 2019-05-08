#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys


data_dir = ['./train.conllu', './dev.conllu', './test.conllu']
for dataset in data_dir:
    idx = []
    word = []
    upos = []
    xpos = []
    head = []
    length = 0
    fout = open(dataset.split('.conllu')[0]+'.new', 'bw')
    with open(dataset, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if line != '':
                line = line.split('\t')
                idx.append(line[0])
                word.append(line[1])
                upos.append(line[3])
                xpos.append(line[4])
                head.append(line[6])
                length += 1
            else:
                fout.write((' '.join(idx)+'\t').encode('utf-8'))
                fout.write((' '.join(word)+'\t').encode('utf-8'))
                fout.write((' '.join(upos)+'\t').encode('utf-8'))
                fout.write((' '.join(xpos)+'\t').encode('utf-8'))
                fout.write((' '.join(head)+'\t').encode('utf-8'))
                fout.write((str(length)+'\n').encode('utf-8'))
                idx = []
                word = []
                upos = []
                xpos = []
                head = []
                length = 0
    fout.close()

