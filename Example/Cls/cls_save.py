import sys
from os import path

import torch
# sys.path.append("..")
# from SequenceClassificationClear import SequenceClassification
from transformers import BertTokenizerFast, BertForSequenceClassification

# 移动到上级目录
# sys.path.append("../../")
# sys.path.append("frameworks")
# torch.jit.save(script, saved_model_path)
# torch.save(model, saved_model_path)
sys.path.append(path.dirname(path.abspath(__file__)))

# from local_dependencies.fun import *


# from bentoml.adapters import DataframeInput

from  PytorchLightingService import PytorchLightingService
svc = PytorchLightingService()

# cls_checkpoint_path="/data_2ch=13-step=11116.ckpt"
model_name="uer/chinese_roberta_L-2_H-128"
tokenizer = BertTokenizerFast.from_pretrained("uer/chinese_roberta_L-2_H-128")
maxLen=64
text="测试分类"
inputData=tokenizer(text,padding="max_length",max_length=maxLen,return_tensors="pt",truncation=True)
# model=SequenceClassification.load_from_checkpoint(checkpoint_path=cls_checkpoint_path)
model=BertForSequenceClassification.from_pretrained(model_name)
# model.load_state_dict(torch.load(cls_checkpoint_path))
model.eval()
# model.freeze()

# script = model.to_torchscript()
saved_model_path = 'model.pt'


# from fun import *


# test()
traced_model = torch.jit.trace(model, (inputData['input_ids'],inputData['token_type_ids'],inputData['attention_mask']))

svc.pack("cls", traced_model)
#保存词典
artifact = {"tokenizer": tokenizer}
svc.pack("tokenizer", artifact)
# # svc.pack('model', saved_model_path)
# svc.set_version("2019-08.iteration20")
svc.save()
# svc.save_to_dir(path="test_out")