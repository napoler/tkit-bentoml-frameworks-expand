import sys
from os import path

import torch

from model import Model

# sys.path.append("..")
# from SequenceClassificationClear import SequenceClassification
# from transformers import BertTokenizerFast, BertForSequenceClassification
# 移动到上级目录
# sys.path.append("../../")
# sys.path.append("frameworks")
# torch.jit.save(script, saved_model_path)
# torch.save(model, saved_model_path)
sys.path.append(path.dirname(path.abspath(__file__)))

# from local_dependencies.fun import *
from transformers import BertTokenizerFast

# from bentoml.adapters import DataframeInput

from PytorchLightingService import SentenceBertService

myService = SentenceBertService()

model_name = "uer/chinese_roberta_L-2_H-128"
tokenizer = BertTokenizerFast.from_pretrained("uer/chinese_roberta_L-2_H-128")
maxLen = 64
text = "测试分类"
inputDataA = tokenizer(text, padding="max_length", max_length=maxLen, return_tensors="pt", truncation=True)
inputDataB = tokenizer(text, padding="max_length", max_length=maxLen, return_tensors="pt", truncation=True)

checkpoint_path = "/mnt/data/dev/github/tkit-bentoml-frameworks-expand/data/checkpoint.ckpt"

model = Model(pretrained="uer/chinese_roberta_L-2_H-512").load_from_checkpoint(checkpoint_path=checkpoint_path)
# model.load_state_dict(torch.load(cls_checkpoint_path))
model.eval()
# model.freeze()

# script = model.to_torchscript()
# saved_model_path = 'model.pt'


# from fun import *


# test()
traced_model = torch.jit.trace(model, (
    inputDataA['input_ids'], inputDataB['input_ids'], inputDataA['attention_mask'], inputDataB['attention_mask']))

myService.pack("model", traced_model)
# 保存词典
artifact = {"tokenizer": tokenizer}
myService.pack("tokenizer", artifact)
# # myService.pack('model', saved_model_path)
# myService.set_version("2019-08.iteration20")
# myService.save()
# myService.save_to_dir(path="test_out")

myService.start_dev_server()

# Stop the dev model server
myService.stop_dev_server()

# Save the entire prediction service to a BentoML bundle
saved_path = myService.save()
