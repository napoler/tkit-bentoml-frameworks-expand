import sys
from os import path

import torch
from transformers import BertTokenizerFast

# sys.path.append("..")
from SequenceClassificationClear import SequenceClassification

# sys.path.append("frameworks")
# torch.jit.save(script, saved_model_path)
# torch.save(model, saved_model_path)
sys.path.append(path.dirname(path.abspath(__file__)))

# from local_dependencies.fun import *


# from bentoml.adapters import DataframeInput

from  bentoml_cls_model import PytorchLightingService
svc = PytorchLightingService()

cls_checkpoint_path="/data_200t/chenyaozu/判断是否多个疾病/18/7a01b14a7f554563bb6f3f52da28c4b4/checkpoints/epoch=13-step=11116.ckpt"
tokenizer = BertTokenizerFast.from_pretrained("/data_200t/chenyaozu/data/base_model/chinese_roberta_L-4_H-512")
maxLen=64
text="高血压，心脏病"
inputData=tokenizer(text,padding="max_length",max_length=maxLen,return_tensors="pt",truncation=True)
model=SequenceClassification.load_from_checkpoint(checkpoint_path=cls_checkpoint_path)
model.eval()
model.freeze()

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