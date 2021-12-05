# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""

import pytorch_lightning as pl
import torch
from torch import nn
from transformers import BertTokenizer, AutoConfig, AutoModel


class Model(pl.LightningModule):
    """



    """

    def __init__(
            self, learning_rate=3e-4,
            T_max=5,
            ignore_index=0, max_len=256,
            optimizer_name="AdamW",
            dropout=0.2,
            labels=2,
            pretrained="uer/chinese_roberta_L-2_H-128",
            batch_size=2,
            trainfile="./data/train.pkt",
            valfile="./data/val.pkt",
            testfile="./data/test.pkt",
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        config = AutoConfig.from_pretrained(pretrained)
        self.model = AutoModel.from_pretrained(pretrained, config=config)
        #         self.rnn = nn.GRU(config.hidden_size, config.hidden_size,dropout=dropout,num_layers=2,bidirectional=True)
        # self.rnn = nn.LSTM(config.hidden_size, config.hidden_size, dropout=dropout, num_layers=2, bidirectional=True)
        self.rnn = nn.GRU(config.hidden_size, config.hidden_size, dropout=dropout, num_layers=2, bidirectional=True)
        self.pre_classifier = nn.Linear(config.hidden_size * 6, config.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)
        # self.classifierSigmoid = torch.nn.Sigmoid()
        #
        # self.tomask = autoMask(
        #     # transformer,
        #     mask_token_id=self.tokenizer.mask_token_id,  # the token id reserved for masking
        #     pad_token_id=self.tokenizer.pad_token_id,  # the token id for padding
        #     mask_prob=0.05,  # masking probability for masked language modeling
        #     replace_prob=0.90,
        #     # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
        #     mask_ignore_token_ids=[self.tokenizer.cls_token_id, self.tokenizer.eos_token_id]
        #     # other tokens to exclude from masking, include the [cls] and [sep] here
        # )

    def forward(self, input_ids_a, input_ids_b, attention_mask_a=None, attention_mask_b=None):
        """
        分类解决方案


        """
        B, L = input_ids_a.size()
        #         print(input_ids_a.size())
        outputs_a = self.model(input_ids=input_ids_a, attention_mask=attention_mask_a)
        # Perform pooling. In this case, max pooling.
        #         emb_a = self.mean_pooling(outputs_a, attention_mask_a)
        emb_a, _ = self.rnn(outputs_a[0])
        #         print(emb_a.size(),emb_a.sum(1).size())
        #         emb_a=emb_a.sum(1).view(B,-1)
        emb_a = self.mean_pooling(emb_a, attention_mask_a)
        #         print(emb_a.size())
        outputs_b = self.model(input_ids=input_ids_b, attention_mask=attention_mask_b)
        emb_b, _ = self.rnn(outputs_b[0])
        # Perform pooling. In this case, max pooling.
        emb_b = self.mean_pooling(emb_b, attention_mask_b)
        #         _,emb_b=self.rnn(outputs_b[0])
        #         emb_b=emb_b.sum(1).view(B,-1)
        emb_diff = emb_a - emb_b

        emb = torch.cat((emb_a, emb_b, emb_diff.abs()), -1)
        pooler = self.pre_classifier(emb)
        #         cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        #         sim=cos(emb_a, emb_b)

        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        # output = self.classifierSigmoid(output)
        return output

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


if __name__ == '__main__':
    pass
