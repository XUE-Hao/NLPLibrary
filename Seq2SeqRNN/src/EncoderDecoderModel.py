# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import random

import src.args as args
import src.utils as utils


class Encoder(nn.Module):
    def __init__(self, input_vocab_size: int, embedding_size: int, encoder_hidden_size: int, decoder_hidden_size: int, num_layers: int, dropout_rate: float):
        super(Encoder, self).__init__()

        self.input_vocab_size = input_vocab_size
        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(self.input_vocab_size, self.embedding_size)  # 输入是index不是one-hot
        self.rnn = nn.GRU(self.embedding_size, self.encoder_hidden_size, num_layers=self.num_layers, bidirectional=True)
        self.linear = nn.Linear(self.encoder_hidden_size * 2, self.decoder_hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, src: Tensor):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        hidden = hidden.permute(1, 0, 2).contiguous().view(-1, self.num_layers, 2 * self.encoder_hidden_size).permute(1, 0, 2)  # 拼接每个batch的两个方向hidden把(num_layers * 2, batch_size, encoder_hidden_size)变成(num_layers, batch_size, 2 * encoder_hidden_size)
        hidden = torch.tanh(self.linear(hidden))
        return outputs, hidden  # hidden(num_layers, batch_size, encoder_hidden_size)


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size: int, encoder_num_layers: int, decoder_hidden_size: int, attention_size: int):
        super(Attention, self).__init__()

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        # self.attention_in_size = self.encoder_hidden_size * 2 + self.decoder_hidden_size * self.encoder_num_layers
        self.attention_size = attention_size

        self.attention = nn.Linear(self.encoder_hidden_size * 2 + self.decoder_hidden_size * self.encoder_num_layers, self.attention_size)

    def forward(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """
        :param decoder_hidden: (num_layers, batch_size, decoder_hidden_size)
        :param encoder_outputs: (src_len, batch_size, 2 * encoder_hidden_size)
        """
        src_len = encoder_outputs.shape[0]
        decoder_hidden = decoder_hidden.permute(1, 0, 2).contiguous().view(-1, 1, self.decoder_hidden_size * self.encoder_num_layers)  # 拼接每个batch的每层hidden把(num_layers, batch_size, encoder_hidden_size)变成(batch_size, 1, num_layers * encoder_hidden_size)
        repeated_encoder_hidden = decoder_hidden.repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attention(torch.cat((repeated_encoder_hidden, encoder_outputs), dim=2)))
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_vocab_size: int, embedding_size: int, encoder_hidden_size: int, encoder_num_layers: int, decoder_hidden_size: int, num_layers: int, attention_size: int, dropout_rate: float):
        super(Decoder, self).__init__()

        self.embedding_size = embedding_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.decoder_hidden_size = decoder_hidden_size
        self.output_vocab_size = output_vocab_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.attention = Attention(encoder_hidden_size, encoder_num_layers, decoder_hidden_size, attention_size)
        self.embedding = nn.Embedding(self.output_vocab_size, self.embedding_size)
        self.rnn = nn.GRU(self.encoder_hidden_size * 2 + self.embedding_size, self.decoder_hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size * 2 + self.embedding_size, self.output_vocab_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _weighted_encoder_representation(self, decoder_hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_representation = torch.bmm(a, encoder_outputs)
        weighted_encoder_representation = weighted_encoder_representation.permute(1, 0, 2)
        return weighted_encoder_representation

    def forward(self, input: Tensor, decoder_hidden: Tensor, encoder_outputs: Tensor):
        """
        :param input: (batch_size)
        """
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        weighted_encoder_representation = self._weighted_encoder_representation(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_encoder_representation), dim=2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden)
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)  # (batch_size, decoder_hidden_size)
        weighted_encoder_representation = weighted_encoder_representation.squeeze(0)
        output = self.out(torch.cat((output, weighted_encoder_representation, embedded), dim=1))
        return output, decoder_hidden


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size: int,
                 output_vocab_size: int,
                 embedding_size: int = args.hidden_size,
                 encoder_hidden_size: int = args.hidden_size,
                 encoder_num_layers: int = args.rnn_num_layers,
                 decoder_hidden_size: int = args.hidden_size,
                 decoder_num_layers: int = args.rnn_num_layers,
                 attention_size: int = args.hidden_size,
                 dropout_rate: float = args.dropout_rate):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_vocab_size, embedding_size, encoder_hidden_size, decoder_hidden_size, encoder_num_layers, dropout_rate)
        self.decoder = Decoder(output_vocab_size, embedding_size, encoder_hidden_size, encoder_num_layers, decoder_hidden_size, decoder_num_layers, attention_size, dropout_rate)

    def forward(self, src: Tensor, trg: Tensor, teach_forcing_ratio: float = 0.) -> Tensor:
        batch_size = src.shape[1]
        max_output_len = trg.shape[0]
        output_vocab_size = self.decoder.output_vocab_size
        outputs = torch.zeros(max_output_len, batch_size, output_vocab_size).to(args.device)
        encoder_outputs, hidden = self.encoder(src)
        output = trg[0, :]  # 第一个必须是<SOS>
        for t in range(1, max_output_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)  # output(batch_size, output_vocabulary.vocab_size)
            outputs[t] = output
            teacher_force = random.random() < teach_forcing_ratio
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)
        return outputs

    def predict(self, src: Tensor):  # src(max_sentence_len, 1)
        """
        :return: 返回生成句子的id列表
        """
        encoder_outputs, hidden = self.encoder(src)
        output = torch.tensor([utils.SOS_id], dtype=torch.long, device=args.device)
        save = [[(1., output, hidden, [utils.SOS_id])], []]  # beam_search用
        res_save = []
        now_i, next_i = 0, 1
        for t in range(1, args.max_sentence_length):
            for p, output, hidden, now_output in save[now_i]:
                output, hidden = self.decoder(output, hidden, encoder_outputs)
                output = F.softmax(output, dim=1)
                for k in range(args.beam_size):
                    top_val, top_id = output.topk(1)
                    top_val = top_val.squeeze(0).item()
                    if top_id.squeeze(0).item() == utils.EOS_id or len(now_output) + 1 == args.max_sentence_length:
                        res_save.append((p * top_val, now_output + [top_id.squeeze(0).item()]))
                    else:
                        save[next_i].append((p * top_val, top_id.squeeze(0), hidden, now_output + [top_id.squeeze(0).item()]))
            save.sort(reverse=True)
            save[next_i] = save[next_i][:args.beam_size]
            save[now_i] = []
            now_i, next_i = next_i, now_i
        return res_save[0][1]


def init_weithts(model: nn.Module):
    for name, parameter in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(parameter.data, mean=0, std=0.01)
        else:
            nn.init.constant_(parameter.data, 0)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
