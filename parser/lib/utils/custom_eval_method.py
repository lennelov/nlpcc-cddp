#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import numpy as np
import sys
import pdb
import json
import re
import logging
import tensorflow as tf
import string
from collections import Counter

logger = logging.getLogger(__name__)


def to_utf8(text):
    if isinstance(text, unicode):
        return text.encode('utf8')
    else:
        return text


def to_unicode(text):
    if isinstance(text, str):
        return text.decode('utf8')
    else:
        return text


def generate_custom_eval_fn(fn_name):
    if fn_name == 'qa':
        return qa_eval
    elif fn_name == 'mrc':
        return mrc_eval
    elif fn_name == 'mrc_old':
        return mrc_eval_old
    elif fn_name == 'bert4binary_qa_all':
        return bert4binary_qa_eval_all
    elif fn_name == 'bert4binary_qa_window':
        return bert4binary_qa_eval_window
    elif fn_name == 'bert4binary_qa_ner':
        return bert4binary_qa_eval_ner
    else:
        raise ValueError('Unknown fn name: %s' % fn_name)


def qa_eval(fetch_result):
    # eval f1
    logits, trues = [], []
    for result in fetch_result:
        logits.append(result['infer'][1])
        trues.append(result['label'])
    return eval_by_pos_cnt(logits, trues)


def mrc_eval(fetch_result):
    answer_result = []
    for result in fetch_result:
        context = result['raw_context']
        context = to_unicode(context)

        span_list = json.loads(result['span_str_list'])
        answer_list = json.loads(result['answer_json'])
        start_pos = span_list[result['yp1']][0]
        end_pos = span_list[result['yp2']][1]

        answer = to_unicode(context[start_pos: end_pos])
        # logger.warn('[MRC_EVAL][answer: {}][answer list: {}]'.format(answer.encode('utf-8'),
        #                                                              result['answer_json'].encode('utf-8')))
        answer_result.append((result['uuid'], result['yp1'], result['yp2'], answer, answer_list))
    return evaluate_em_f1(answer_result)


def mrc_eval_old(fetch_result):
    answer_result = []
    for result in fetch_result:
        context = to_unicode(result['context_tokens'])
        context_segs = context.split(u' ')
        answer_list = json.loads(result['answer_json'])
        answer = u' '.join(context_segs[result['yp1']: result['yp2'] + 1])
        # logger.warn('[MRC_EVAL][answer: {}][answer list: {}]'.format(answer.encode('utf-8'),
        #                                                              result['answer_json'].encode('utf-8')))
        answer_result.append((result['uuid'], result['yp1'], result['yp2'], answer, answer_list))
    return evaluate_em_f1(answer_result)


def bert4binary_qa_eval_all(fetch_result):
    """
    ner和window两个层面
    """
    window_logit_list, window_true_list = [], []
    ner_logit_list, ner_true_list = [], []
    for result in fetch_result:
        ner_label = result['ner_label']
        ner_mask = result['ner_mask']
        probs = result['start_probs']
        for mask, prob, label in zip(ner_mask, probs, ner_label):
            if mask == 1:
                ner_logit_list.append(prob)
                ner_true_list.append(label)

        masked_probs = [x * y for x, y in zip(ner_mask, probs)]
        max_prob = max(masked_probs)
        true_label = max(ner_label)
        window_logit_list.append(max_prob)
        window_true_list.append(true_label)
        # logging的方式的打印相关的信息
        debug_str = '%s\n%s\n%s\n%s\n%s\n%s\n' % (result['uuid'], result['answer'], result['input_str'],
                                                  result['raw_input'], ' '.join(list(map(str, masked_probs))),
                                                  ' '.join(list(map(str, ner_label))))
        logging.warn(debug_str)

    metrics_dict = {}
    window_eval_str, window_metrics_dict = eval_by_pos_cnt(window_logit_list, window_true_list)
    ner_eval_str, ner_metrics_dict = eval_by_pos_cnt(ner_logit_list, ner_true_list)
    for key, value in window_metrics_dict.items():
        metrics_dict['window_%s' % key] = value
    for key, value in ner_metrics_dict.items():
        metrics_dict['ner_%s' % key] = value
    eval_str = '{WINDOW}%s{NER}%s' % (window_eval_str, ner_eval_str)
    return eval_str, metrics_dict


def bert4binary_qa_eval_ner(fetch_result):
    """
    只计算NER的指标
    """
    logit_list, true_list = [], []
    for result in fetch_result:
        ner_label = result['ner_label']
        ner_mask = result['ner_mask']
        probs = result['start_probs']
        for mask, prob, label in zip(ner_mask, probs, ner_label):
            if mask == 1:
                logit_list.append(prob)
                true_list.append(label)
        masked_probs = [x * y for x, y in zip(ner_mask, probs)]
        # logging的方式的打印相关的信息
        debug_str = '%s\n%s\n%s\n%s\n%s\n%s\n' % (result['uuid'], result['answer'], result['input_str'],
                                                  result['raw_input'], ' '.join(list(map(str, masked_probs))),
                                                  ' '.join(list(map(str, ner_label))))
        logging.warn(debug_str)
    return eval_by_pos_cnt(logit_list, true_list)


def bert4binary_qa_eval_window(fetch_result):
    """
    每个NER位置二分类, 评测指标 p r f1
    每个记录选取prob最高的NER作为预测, 以及它对应的NER label
    """
    logit_list, true_list = [], []
    for result in fetch_result:
        ner_label = result['ner_label']
        ner_mask = result['ner_mask']
        probs = result['start_probs']
        masked_probs = [x * y for x, y in zip(ner_mask, probs)]
        max_prob = max(masked_probs)
        true_label = max(ner_label)
        logit_list.append(max_prob)
        true_list.append(true_label)
        # logging的方式的打印相关的信息
        debug_str = '%s\n%s\n%s\n%s\n%s\n%s\n' % (result['uuid'], result['answer'], result['input_str'],
                                                result['raw_input'], ' '.join(list(map(str, masked_probs))),
                                                ' '.join(list(map(str, ner_label))))
        logging.warn(debug_str)
    return eval_by_pos_cnt(logit_list, true_list)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_em_f1(answer_result):
    f1 = exact_match = total = 0
    for uuid, yp1, yp2, prediction, ground_truths in answer_result:
        total += 1
        em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1_single = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        exact_match += em
        f1 += f1_single
        logger.warn('[%s][%d,%d][em:%.4f][f1:%.4f][answer: %s][answer list: %s]' %
                    (uuid, yp1, yp2, em, f1_single, prediction.encode('utf-8'),
                     json.dumps(ground_truths, encoding='utf-8')))
    exact_match = float(exact_match) / total
    f1 = float(f1) / total
    eval_str = "{exact match:%.4f}{f1:%.4f}" % (exact_match, f1)
    metrics_dict = {
        'em': exact_match,
        'f1': f1
    }
    return eval_str, metrics_dict


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval_by_pos_cnt(logits, trues):
    pred = [1 if x > 0.5 else 0 for x in logits]
    f1 = metrics.f1_score(y_pred=pred, y_true=trues)
    p = metrics.precision_score(y_pred=pred, y_true=trues)
    r = metrics.recall_score(y_pred=pred, y_true=trues)

    pos_cnt = int(sum(trues))
    data = zip(logits, trues)
    data.sort(key=lambda x: x[0], reverse=True)
    _, trues = zip(*data)
    pred = [1] * pos_cnt + [0] * (len(trues) - pos_cnt)
    eval_f1 = metrics.f1_score(y_pred=pred, y_true=trues)
    metrics_dict = {
        'p': p,
        'r': r,
        'f1': f1,
        'eval': eval_f1
    }
    eval_str = "{custom f1:%.4f}{custom p:%.4f}{custom r:%.4f}{custom eval:%.4f}" % (f1, p, r, eval_f1)
    return eval_str, metrics_dict