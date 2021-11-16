# -*- encoding: utf-8 -*-

import src.EncoderDecoderModel as EncoderDecoderModel
from src.train import *

model = EncoderDecoderModel.Seq2Seq(utils.vocabulary.vocab_size, utils.vocabulary.vocab_size, is_train=False).to(args.device)
model.load_state_dict(torch.load(args.ckpt_dir + 'best_model.ckpt'))
model.eval()


def predict(source):
    source, real_length = utils.sentence2ids(source)
    source = source[:real_length]
    source = torch.tensor(source, dtype=torch.long, device=args.device).view(-1, 1)
    with torch.no_grad():
        return model.predict(source)[0]


if __name__ == '__main__':
    while True:
        a = input()
        ans = predict(a)
        for id in ans:
            print(utils.vocabulary.index2word[id.item()] + ' ', end='')
        print('')
    # ans = predict('看过《我是山姆》吗？')
    # for id in ans:
    #     print(utils.output_vocabulary.index2word[id] + ' ', end='')
    # print('')