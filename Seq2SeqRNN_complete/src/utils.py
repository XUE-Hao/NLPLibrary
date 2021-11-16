import torch
import json
import jieba
import random

import src.args as args

PAD_id = 0
SOS_id = 1
EOS_id = 2
UNK_id = 3

dir_path = "../data/film/"  # 由于每个领域知识图谱是分开的，所以需要对每个领域单独处理
file_names = ("train.json", "dev.json", "test.json")


class Vocabulary:
    def __init__(self):
        self.word2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.word2count = {'<PAD>': 0, '<SOS>': 0, '<EOS>': 0, '<UNK>': 0}
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
        for word in jieba.lcut(sentence):  # 根据语言选择分词方式
            self.add_word(word)

    def remove_low_frequency_words(self, remove_frequency=args.remove_frequency):
        print("原始词表大小：", self.vocab_size)
        num_low_frequency_word = 0
        for i in range(4, self.vocab_size):
            word = self.index2word[i]
            if self.word2count[word] <= remove_frequency:
                num_low_frequency_word += 1
                del self.word2index[word]
                del self.word2count[word]
                del self.index2word[i]
            else:
                self.word2index[word] = i - num_low_frequency_word
                del self.index2word[i]
                self.index2word[i - num_low_frequency_word] = word
        self.vocab_size -= num_low_frequency_word
        print("去除低频词后词表大小", self.vocab_size)

    def get_word_id(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return UNK_id


def prepare_data() -> (Vocabulary, [[(str, str)]]):
    """
    读取预处理后的数据文件。构建词表，并保存数据集中的数据
    :return:
    """
    vocabulary = Vocabulary()
    datasets = []
    for file_name in file_names:
        with open(dir_path + "processed_" + file_name, 'r', encoding='utf-8') as in_file:
            in_file = json.load(in_file)
            for pair in in_file:
                vocabulary.add_sentence(pair[0])
                vocabulary.add_sentence(pair[1])
            datasets.append(in_file)
    vocabulary.remove_low_frequency_words()
    return vocabulary, datasets


vocabulary, datasets = prepare_data()


def sentence2ids(sentence: str) -> ([int], int):
    indexes = [SOS_id]
    for i, word in enumerate(jieba.lcut(sentence)):  # 需要根据语言修改
        if i >= args.max_sentence_length - 2:
            break
        indexes.append(vocabulary.get_word_id(word))
    indexes.append(EOS_id)
    real_length = len(indexes)
    for i in range(args.max_sentence_length - len(indexes)):
        indexes.append(PAD_id)
    return indexes, real_length


def batch_data2tensor(data: [(str, str)], dataset_type: str):
    src_save = [[] for _ in range(args.max_sentence_length)]
    target_save = [[] for _ in range(args.max_sentence_length)]
    now_batch_src_max_length, now_batch_target_max_length = 0, 0
    for i, pair in enumerate(data):
        src_ids, src_real_length = sentence2ids(pair[0])
        target_ids, target_real_length = sentence2ids(pair[1])
        now_batch_src_max_length = max(now_batch_src_max_length, src_real_length)
        now_batch_target_max_length = max(now_batch_target_max_length, target_real_length)
        for j in range(args.max_sentence_length):
            src_save[j].append(src_ids[j])
            target_save[j].append(target_ids[j])
    src_save = src_save[:now_batch_src_max_length]
    if dataset_type == 'train':
        target_save = target_save[:now_batch_target_max_length]  # 不然长度会和预测出来的不一致
    return torch.tensor(src_save, dtype=torch.long, device=args.device), torch.tensor(target_save, dtype=torch.long, device=args.device)


def get_batch_data(pairs: [(str, str)], dataset_type: str):
    if dataset_type == 'train':
        random.shuffle(pairs)
    batch_data = []
    for start_id in range(0, len(pairs), args.batch_size):
        end_id = min(start_id + args.batch_size, len(pairs))
        batch_data.append(batch_data2tensor(pairs[start_id: end_id], dataset_type))
    return batch_data


# train_dataset = get_batch_data(datasets[0], dataset_type='train')
dev_dataset = get_batch_data(datasets[1], dataset_type='dev')
test_dataset = get_batch_data(datasets[2], dataset_type='test')
