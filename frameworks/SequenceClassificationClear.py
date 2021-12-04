# -*- coding: utf-8 -*-

"""
SequenceClassification.py
用于序列分类任务

"""

import pytorch_lightning as pl
from transformers import BertTokenizer, AutoConfig, BertForSequenceClassification


# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# loss = outputs.loss
# logits = outputs.logits



class SequenceClassification(pl.LightningModule):
    """
    基础的命名实体


    """

    def __init__(
        self, learning_rate=3e-4, T_max=5,
        optimizer_name="AdamW", dropout=0.2, pretrained="uer/chinese_roberta_L-2_H-128",num_labels=2,
        batch_size=2, trainfile="./data/train.pkt", valfile="./data/val.pkt", testfile="./data/test.pkt", **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        config = AutoConfig.from_pretrained(pretrained,torchscript=True)
        # config.num_labels=self.hparams.max_len
        # config.output_attentions = True
        # self.hparams.config = config
        config.num_labels=num_labels
        config.problem_type="single_label_classification"
        # self.model = BertForPreTraining.from_pretrained(pretrained, config=config)
        self.model = BertForSequenceClassification.from_pretrained(pretrained, config=config)
    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.model(input_ids, token_type_ids=token_type_ids,
                       attention_mask=attention_mask)

        return outputs
