# -*- encoding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import torch
from io import open
import unicodedata
import random
import re

import src.args as args

PAD_id = 0
SOS_id = 1
EOS_id = 2
UNK_id = 3


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.vocab_size = 4  # 初始为特殊标签的个数

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.vocab_size
            self.word2count[word] = 1
            self.index2word[self.vocab_size] = word
            self.vocab_size += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence):
        for word in sentence.split(' '):  # TODO: 中文需要换分词方式
            self.add_word(word)


# 把一个Unicode字符串转换成纯ASCII
def unicode2ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# 转换成小写、去除标点、去除非字母符号
def normalize_string(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_languages(language1, language2, reverse=False):
    print('读取数据...')
    # 读文件并划分成行
    lines = open('../data/%s-%s.txt' % (language1, language2), encoding='utf-8').read().strip().split('\n')
    # 把每行划分成对，并规范化
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_vocabulary = Vocabulary(language2)
        output_vocabulary = Vocabulary(language1)
    else:
        input_vocabulary = Vocabulary(language1)
        output_vocabulary = Vocabulary(language2)

    return input_vocabulary, output_vocabulary, pairs


def prepareData(lang1, lang2, reverse=False):
    input_vocabulary, output_vocabulary, pairs = read_languages(lang1, lang2, reverse)
    print("读取 %s 个句子对" % len(pairs))
    for pair in pairs:
        input_vocabulary.add_sentence(pair[0])
        output_vocabulary.add_sentence(pair[1])
    print("统计词表大小:")
    print(input_vocabulary.name, input_vocabulary.vocab_size)
    print(output_vocabulary.name, output_vocabulary.vocab_size)
    return input_vocabulary, output_vocabulary, pairs


input_vocabulary, output_vocabulary, pairs = prepareData('eng', 'fra', True)
random.shuffle(pairs)
train_pairs = pairs[0: len(pairs) * 8 // 10]
valid_pairs = pairs[len(pairs) * 8 // 10: len(pairs) * 9 // 10]
test_pairs = pairs[len(pairs) * 9 // 10: len(pairs)]


def sentence2ids(language, sentence):
    indexes = [SOS_id]
    for i, word in enumerate(sentence.split(' ')):  # TODO: 中文需要修改
        if i >= args.max_sentence_length - 2:
            break
        if word in language.word2index:
            indexes.append(language.word2index[word])
        else:
            indexes.append(UNK_id)
    indexes.append(EOS_id)
    real_length = len(indexes)
    for i in range(args.max_sentence_length - len(indexes)):
        indexes.append(PAD_id)
    return indexes, real_length


def batch_data2tensor(data):
    src_save = [[] for _ in range(args.max_sentence_length)]
    target_save = [[] for _ in range(args.max_sentence_length)]
    now_batch_src_max_length, now_batch_target_max_length = 0, 0
    for i, pair in enumerate(data):
        src_ids, src_real_length = sentence2ids(input_vocabulary, pair[0])
        target_ids, target_real_length = sentence2ids(output_vocabulary, pair[1])
        now_batch_src_max_length = max(now_batch_src_max_length, src_real_length)
        now_batch_target_max_length = max(now_batch_target_max_length, target_real_length)
        for j in range(args.max_sentence_length):
            src_save[j].append(src_ids[j])
            target_save[j].append(target_ids[j])
    src_save = src_save[:now_batch_src_max_length]
    target_save = target_save[:now_batch_target_max_length]
    return torch.tensor(src_save, dtype=torch.long, device=args.device), torch.tensor(target_save, dtype=torch.long, device=args.device)


def get_batch_data(pairs):
    batch_data = []
    for start_id in range(0, len(pairs), args.batch_size):
        end_id = min(start_id + args.batch_size, len(pairs))
        batch_data.append(batch_data2tensor(pairs[start_id: end_id]))
    return batch_data


train_dataset = get_batch_data(train_pairs)
valid_dataset = get_batch_data(valid_pairs)
test_dataset = get_batch_data(test_pairs)
