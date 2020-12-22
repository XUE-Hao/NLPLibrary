# -*- encoding: utf-8 -*-

import torch
import numpy as np

hidden_size = 512
batch_size = 32
learning_rate = 0.001
epochs = 30
data_path = './data/'
ckpt_dir = '../model/'
embedding_size = 200
max_sentence_length = 20
dropout_rate = 0.5
teach_forcing_ratio = 0.5
log_batch_interval = 100  # 多少个batch记录一次
clip = 1  # torch.nn.utils.clip_grad_norm_的参数
rnn_num_layers = 3
beam_size = 5
rand_seed = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)
np.random.seed(rand_seed)
torch.backends.cudnn.deterministic = True
