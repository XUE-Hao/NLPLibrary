import torch.nn as nn
from transformers import BertModel, BertConfig
import src.args as args


class KnowledgeSelector(nn.Module):
    def __init__(self):
        super(KnowledgeSelector, self).__init__()
        self.bert_model = BertModel.from_pretrained(args.bert_pretrained_model_name)
        self.cls_classifier = nn.Linear(args.bert_hidden_size, args.num_labels)
        for p in self.bert_model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, args.bert_hidden_size]  TODO: 原代码取倒数第二层的输出值作为句向量
        cls_states = last_hidden_states.transpose(0, 1)[0]  # [batch_size, args.bert_hidden_size]
        return self.cls_classifier(cls_states)  # [batch_size, args.num_labels]
