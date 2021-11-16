# -*- encoding: utf-8 -*-

import torch
from torch import optim
import time
import matplotlib
import matplotlib.pyplot as plt
import random
import math
import numpy as np

import src.args as args
import src.EncoderDecoderModel as EncoderDecoderModel
import src.utils as utils
import src.metrics as metrics

model = EncoderDecoderModel.Seq2Seq(utils.vocabulary.vocab_size, utils.vocabulary.vocab_size).to(args.device)
print(f'此模型包含 {EncoderDecoderModel.count_parameters(model):,} 个可训练参数')
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
criterion = torch.nn.CrossEntropyLoss(ignore_index=utils.PAD_id)  # CrossEntropyLoss处理的是没有经过softmax的值
all_train_losses = []
all_valid_losses = []
all_bleu1 = []
all_bleu2 = []
batch_num_train_dataset = -1


def train_epoch():  # 训练一个epoch
    model.train()
    total_loss = 0.
    start_time = time.time()
    train_dataset = utils.get_batch_data(utils.datasets[0], dataset_type='train')
    global batch_num_train_dataset
    batch_num_train_dataset = len(train_dataset)
    for batch_id in range(batch_num_train_dataset):
        source, target = train_dataset[batch_id]
        optimizer.zero_grad()
        output = model(source, target, args.teach_forcing_ratio)
        output = output[1:].view(-1, output.shape[-1])
        target = target[1:].view(-1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        if batch_id and batch_id % args.log_batch_interval == 0:
            batch_loss = total_loss / args.log_batch_interval
            all_train_losses.append(batch_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:f} | ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, batch_id, batch_num_train_dataset, args.learning_rate, elapsed * 1000 / args.log_batch_interval,
                batch_loss))
            total_loss = 0
            start_time = time.time()


def evaluate():
    model.eval()  # 开启评测模式
    total_loss = 0.
    total_bleu1 = 0.
    total_bleu2 = 0.
    total_data = 0
    with torch.no_grad():
        for source, target in utils.dev_dataset:
            total_data += target.size(1)
            res, output = model.predict(source)
            res = res[1:]
            output = output[1:]
            target = target[1:]
            # 计算BLEU
            bleu1, bleu2 = metrics.bleu(res, target)
            total_bleu1 += np.sum(bleu1)
            total_bleu2 += np.sum(bleu2)
            # 计算损失
            output = output.view(-1, output.shape[-1])
            target = target.view(-1)
            loss = criterion(output, target)
            total_loss += loss
    loss = total_loss / len(utils.dev_dataset)
    bleu1 = total_bleu1 / total_data
    bleu2 = total_bleu2 / total_data
    return loss, math.exp(loss), bleu1, bleu2  # 只有使用交叉熵的时候才可以这样算ppl


if __name__ == '__main__':
    best_bleu2 = float('inf')

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_epoch()
        val_loss, ppl, bleu1, bleu2 = evaluate()
        all_valid_losses.append(val_loss)
        all_bleu1.append(bleu1)
        all_bleu2.append(bleu2)
        print('-' * 89)
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.2f} | ppl: {:5.2f} | bleu1: {:5.2f} | bleu2: {:5.2f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, ppl, bleu1, bleu2))
        print('-' * 89)

        torch.save(model.state_dict(), args.ckpt_dir + 'model_epoch_' + epoch.__str__() + '.ckpt')
        if bleu2 > best_bleu2:
            best_bleu2 = bleu2
            torch.save(model.state_dict(), args.ckpt_dir + 'best_model.ckpt')
    plt.figure()
    plt.title("loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.plot([i for i in range(1, args.epochs * (batch_num_train_dataset // args.log_batch_interval) + 1)], all_train_losses)
    plt.plot([i * (batch_num_train_dataset // args.log_batch_interval) for i in range(1, args.epochs + 1)], all_valid_losses)
    plt.show()
    plt.figure()
    plt.title("bleu")
    plt.xlabel("epoch")
    plt.ylabel("bleu")
    plt.plot(all_bleu1)
    plt.plot(all_bleu2)
    plt.show()
