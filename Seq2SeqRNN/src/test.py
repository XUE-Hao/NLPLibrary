# -*- encoding: utf-8 -*-

from src.train import *

model = Seq2Seq(utils.input_vocabulary.vocab_size, utils.output_vocabulary.vocab_size).to(args.device)
model.load_state_dict(torch.load(args.ckpt_dir + 'best_model.ckpt'))
model.eval()


def predict(source):
    source, real_length = utils.sentence2ids(utils.input_vocabulary, source)
    source = source[:real_length]
    source = torch.tensor(source, dtype=torch.long, device=args.device).view(-1, 1)
    with torch.no_grad():
        return model.predict(source)


if __name__ == '__main__':
    while True:
        a = input()
        ans = predict(a)
        for id in ans:
            print(utils.output_vocabulary.index2word[id] + ' ', end='')
        print('')
    # ans = predict('Sois un homme !')
    # for id in ans:
    #     print(utils.output_vocabulary.index2word[id] + ' ', end='')
    # print('')
