# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""

import pytorch_lightning as pl
from transformers import BertTokenizer, AutoModelForTokenClassification, AutoConfig


class ClsModel(pl.LightningModule):
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
        self.model = AutoModelForTokenClassification.from_pretrained(pretrained, config=config)

    def forward(self, input_ids,  attention_mask=None):
        """
        分类解决方案
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output[0]


if __name__ == '__main__':
    pass
