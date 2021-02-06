import torch

num_epochs = 10
batch_size = 32
num_layers = 2
max_sentence_length = 100
warm_up_epoches = 5
log_batch_interval = 100  # 多少个batch记录一次
ckpt_dir = '../model/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 5000
num_batchs = 400
