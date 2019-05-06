#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import sys
import logging

logger = logging.getLogger(__name__)

ndcg_list = [1, 3, 5, 7, 10]

# result列表，每一行是一个Q-U pair。对其进行按query合并，然后计算指标
def output_eval_results(results):
    hash_dict = {}
    count = 0
    total_count = 0
    query_scores, query_labels = [], []
    all_query_texts = []
    all_title_texts = []
    for result in results:
        score = result['infer'][0]
        label = result['eval_label']
        x_word_raw = result['x_word_raw']
        y_word_raw = result['y_word_raw']
        hash_id = result['hash_id']
        total_count += 1
        if hash_id not in hash_dict:
            hash_dict[hash_id] = id = count
            count += 1
            query_labels.append([])
            query_scores.append([])
            all_title_texts.append([])
            all_query_texts.append(x_word_raw)
        else:
            id = hash_dict[hash_id]

        all_title_texts[id].append(y_word_raw)
        query_scores[id].append(score)
        query_labels[id].append(label)

    ndcg_dict = ndcg(query_scores, query_labels, True)
    avg_acc = accuracy(query_scores, query_labels)
    return ndcg_dict, avg_acc


def accuracy(query_scores, query_labels):
    assert len(query_scores) == len(query_labels)

    avg_acc = 0
    num_valid_query = 0
    for q in range(len(query_scores)):
        num_title = len(query_scores[q])
        index_ = range(num_title)
        random.shuffle(index_)
        scores = [query_scores[q][i] for i in index_]
        labels = [query_labels[q][i] for i in index_]

        acc = 0
        count = 0
        for i in xrange(num_title):
            for j in xrange(i + 1, num_title):
                if labels[i] != labels[j]:
                    count += 1
                    if scores[i] == scores[j]:
                        acc += 0.5
                    elif (scores[i] - scores[j]) * (labels[i] - labels[j]) > 0:
                        acc += 1

        if count == 0:
            continue
        num_valid_query += 1
        acc = acc * 1.0 / count
        avg_acc += acc
    if num_valid_query == 0:
        logger.warn("valid_num_query = 0")
        return 0.
    return round(avg_acc * 1.0 / num_valid_query, 5)


# scores: 一行一整个query，一行包含多个title
def ndcg(query_scores, query_labels, ndcg_each_query=False):
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def ndcg_at_k(r, k):
        idcg = dcg_at_k(sorted(r, reverse=True), k)
        if not idcg:
            return 0.
        return dcg_at_k(r, k) / idcg

    ndcg_dict = {}
    for n in ndcg_list:
        ndcg_dict[n] = {
            'avg': 0.0,
            'all': [],
        }

    assert len(query_scores) == len(query_labels)
    num_query = len(query_scores)
    for i in range(num_query):
        num_title = len(query_scores[i])
        index_ = range(num_title)
        random.shuffle(index_)
        scores = [query_scores[i][j] for j in index_]
        labels = [query_labels[i][j] for j in index_]
        new_rank = sorted(zip(scores, range(len(scores))), key=lambda x: x[0], reverse=True)

        new_labels = []
        for s, i in new_rank:
            new_labels.append(labels[i])

        for n in ndcg_list:
            ndcg_n = ndcg_at_k(new_labels, n)
            if ndcg_each_query:
                ndcg_dict[n]['all'].append(ndcg_n)

            ndcg_dict[n]['avg'] += ndcg_n

    for n in ndcg_list:
        ndcg_dict[n]['avg'] = round(ndcg_dict[n]['avg'] * 1.0 / num_query, 5) if num_query else 0

    return ndcg_dict
