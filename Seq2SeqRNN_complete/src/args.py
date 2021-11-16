import torch

hidden_size = 512
batch_size = 32
learning_rate = 0.001
epochs = 30
data_path = './data/'
ckpt_dir = '../ckpt/'
embedding_size = 200
max_sentence_length = 50
dropout_rate = 0.5
teach_forcing_ratio = 0.5
log_batch_interval = 100  # 多少个batch记录一次
clip = 1  # torch.nn.utils.clip_grad_norm_的参数
rnn_num_layers = 3
beam_size = 1
rand_seed = 1
tencent_embedding_path = "../Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt"
remove_frequency = 1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
