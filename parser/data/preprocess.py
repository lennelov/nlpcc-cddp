#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Get POS2id table
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
for item in pos:
    fout.write(item+'\n')
fout.close()

# Get word2id table
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

fout = open('./word2id.random.table', 'bw')
word = set()
for dataset in data_dir:
    if dataset.startswith('./train'):
        with open(dataset, encoding='utf-8') as fin:
            for line in fin:
                line = line.strip()
                if line != '':
                    line = line.split('\t')
                    word.add(line[1])
word = list(word)
for item in word:
    fout.write((item+'\n').encode('utf-8'))
fout.close()

max_len = 0
# Transform to the form framework supports
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
                max_len = length if length > max_len else max_len
                idx = []
                word = []
                upos = []
                xpos = []
                head = []
                length = 0
    fout.close()

print(max_len)
