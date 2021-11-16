# -*- encoding: utf-8 -*-

from torch import optim
import time
import matplotlib.pyplot as plt
import random

from src.EncoderDecoderModel import *
import src.utils as utils

model = Seq2Seq(utils.input_vocabulary.vocab_size, utils.output_vocabulary.vocab_size).to(args.device)
model.apply(init_weithts)
print(f'此模型包含 {count_parameters(model):,} 个可训练参数')
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
criterion = nn.CrossEntropyLoss(ignore_index=utils.PAD_id)  # CrossEntropyLoss处理的是没有经过softmax的值
all_train_losses = []
all_valid_losses = []


def train():  # 训练一个epoch
    model.train()
    total_loss = 0.
    start_time = time.time()
    for batch_id in range(len(utils.train_dataset)):
        source, target = utils.train_dataset[batch_id]
        optimizer.zero_grad()
        output = model(source, target, args.teach_forcing_ratio)
        output = output[1:].view(-1, output.shape[-1])
        target = target[1:].view(-1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        if batch_id % args.log_batch_interval == 0:
            batch_loss = total_loss / args.log_batch_interval
            all_train_losses.append(batch_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:f} | ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, batch_id, len(utils.train_dataset), args.learning_rate,
                              elapsed * 1000 / args.log_batch_interval, batch_loss))
            total_loss = 0
            start_time = time.time()


def evaluate():
    model.eval()  # 开启评测模式
    total_loss = 0.
    with torch.no_grad():
        for source, target in utils.valid_dataset:
            output = model(source, target, 0)  # 使用自己的输出作为下个时刻的输入
            output = output[1:].view(-1, output.shape[-1])
            target = target[1:].view(-1)
            loss = criterion(output, target)
            total_loss += loss
    return total_loss / len(utils.valid_dataset)


if __name__ == '__main__':
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate()
        all_valid_losses.append(val_loss)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss))
        print('-' * 89)

        torch.save(model.state_dict(), args.ckpt_dir + 'model_epoch_' + epoch.__str__() + '.ckpt')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    torch.save(best_model.state_dict(), args.ckpt_dir + 'best_model.ckpt')
    plt.figure()
    plt.plot([i for i in range(1, args.epochs * (len(utils.train_dataset) // args.log_batch_interval) + 1)], all_train_losses)
    plt.plot([i * (len(utils.train_dataset) // args.log_batch_interval) for i in range(1, args.epochs + 1)], all_valid_losses)
    plt.show()
