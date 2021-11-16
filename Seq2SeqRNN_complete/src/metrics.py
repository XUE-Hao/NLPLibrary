#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/metrics.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

import args
import utils


def accuracy(logits, targets, padding_idx=None):  # 还没根据自己写法调整
    """
    logits: (batch_size, max_len, vocab_size)
    targets: (batch_size, max_len)
    """
    _, preds = logits.max(dim=2)
    trues = (preds == targets).float()
    if padding_idx is not None:
        weights = targets.ne(padding_idx).float()
        acc = (weights * trues).sum(dim=1) / weights.sum(dim=1)
    else:
        acc = trues.mean(dim=1)
    acc = acc.mean()
    return acc


def attn_accuracy(logits, targets):  # 还没根据自己写法调整
    """
    logits: (batch_size, vocab_size)
    targets: (batch_size)
    """
    _, preds = logits.squeeze(1).max(dim=-1)
    trues = (preds == targets).float()
    acc = trues.mean()
    return acc


def bleu(hyps: Tensor, refs: Tensor):
    """
    hyps(sentence_length, batch_size)
    refs(sentence_length, batch_size)
    """
    # 数据处理
    batch_size = refs.size(1)
    predicts = [''] * batch_size
    targets = [''] * batch_size
    for batch_id, batch_tensor in enumerate(hyps.permute(1, 0)):
        for pos_id, word_id in enumerate(batch_tensor):
            if word_id == utils.EOS_id:
                break
            predicts[batch_id] += utils.vocabulary.index2word[word_id.item()]
    for batch_id, pos_tensor in enumerate(refs.permute(1, 0)):
        for pos_id, word_id in enumerate(pos_tensor):
            if word_id == utils.EOS_id:
                break
            targets[batch_id] += utils.vocabulary.index2word[word_id.item()]
    # 计算BLEU
    bleu_1 = []
    bleu_2 = []
    for predict, target in zip(predicts, targets):
        print("predict:", predict)
        print("target:", target)
        try:
            score = bleu_score.sentence_bleu(
                [target], predict,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
            print("bleu1", score)
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [target], predict,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
            print("bleu2", score)
        except:
            score = 0
        bleu_2.append(score)
    return bleu_1, bleu_2


def distinct(seqs):  # 还没根据自己写法调整
    """
    distinct
    """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


def cosine(X, Y):  # 还没根据自己写法调整
    """
    cosine
    """
    sim = np.sum(X * Y, axis=1) / \
        (np.sqrt((np.sum(X * X, axis=1) * np.sum(Y * Y, axis=1))) + 1e-10)
    return sim


class EmbeddingMetrics(object):  # 还没根据自己写法调整
    """
    EmbeddingMetrics
    """
    def __init__(self, field):
        self.field = field
        assert field.embeddings is not None
        self.embeddings = np.array(field.embeddings)

    def texts2embeds(self, texts):
        """
        texts2embeds
        """
        texts = [self.field.numericalize(text)[1:-1] for text in texts]
        embeds = []
        for text in texts:
            vecs = self.embeddings[text]
            mask = vecs.any(axis=1)
            vecs = vecs[mask]
            if vecs.shape[0] == 0:
                vecs = np.zeros((1,) + vecs.shape[1:])
            embeds.append(vecs)
        return embeds

    def average(self, embeds):
        """
        average
        """
        avg_embeds = [embed.mean(axis=0) for embed in embeds]
        avg_embeds = np.array(avg_embeds)
        return avg_embeds

    def extrema(self, embeds):
        """
        extrema
        """
        ext_embeds = []
        for embed in embeds:
            s_max = np.max(embed, axis=0)
            s_min = np.min(embed, axis=0)
            s_plus = np.abs(s_min) <= s_max
            s = s_max * s_plus + s_min * np.logical_not(s_plus)
            ext_embeds.append(s)
        ext_embeds = np.array(ext_embeds)
        return ext_embeds

    def greedy(self, hyp_embeds, ref_embeds):
        """
        greedy
        """
        greedy_sim = []
        for hyp_embed, ref_embed in zip(hyp_embeds, ref_embeds):
            cos_sim = cosine_similarity(hyp_embed, ref_embed)
            g_sim = (cos_sim.max(axis=1).mean() +
                     cos_sim.max(axis=0).mean()) / 2
            greedy_sim.append(g_sim)
        greedy_sim = np.array(greedy_sim)
        return greedy_sim

    def embed_sim(self, hyp_texts, ref_texts):
        """
        embed_sim
        """
        assert len(hyp_texts) == len(ref_texts)
        hyp_embeds = self.texts2embeds(hyp_texts)
        ref_embeds = self.texts2embeds(ref_texts)

        ext_hyp_embeds = self.extrema(hyp_embeds)
        ext_ref_embeds = self.extrema(ref_embeds)
        ext_sim = cosine(ext_hyp_embeds, ext_ref_embeds)
        # print(len(ext_sim), (ext_sim > 0).sum())
        # print(ext_sim.sum() / (ext_sim > 0).sum())
        ext_sim_avg = ext_sim.mean()

        avg_hyp_embeds = self.average(hyp_embeds)
        avg_ref_embeds = self.average(ref_embeds)
        avg_sim = cosine(avg_hyp_embeds, avg_ref_embeds)
        # print(len(avg_sim), (avg_sim > 0).sum())
        # print(avg_sim.sum() / (avg_sim > 0).sum())
        avg_sim_avg = avg_sim.mean()

        greedy_sim = self.greedy(hyp_embeds, ref_embeds)
        # print(len(greedy_sim), (greedy_sim > 0).sum())
        # print(greedy_sim.sum() / (greedy_sim > 0).sum())
        greedy_sim_avg = greedy_sim.mean()

        return ext_sim_avg, avg_sim_avg, greedy_sim_avg
