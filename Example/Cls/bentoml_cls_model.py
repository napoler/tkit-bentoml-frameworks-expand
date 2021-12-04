# from model.SequenceClassificationClear import SequenceClassification

import bentoml
from bentoml.adapters import JsonInput, JsonOutput
# from bentoml.adapters import DataframeInput
from bentoml.frameworks.pytorch import PytorchModelArtifact

from TokenizerArtifact import TokenizerArtifact


# import bentoml
# from bentoml.adapters import JsonInput

# import sys
# from os import path
# sys.path.append(path.dirname(path.abspath(__file__)))

# from local_dependencies.fun import *














# @bentoml.env(pip_packages=["transformers==4.12.5"])
# @bentoml.artifacts([TokenizerArtifact("tokenizer")])
# class TokenizerService(bentoml.BentoService):
#     @bentoml.api(input=JsonInput(), batch=False)
#     def predict(self, parsed_json):
#         src_text = parsed_json.get("text")
#         # model = self.artifacts.gptModel.get("model")
#         tokenizer = self.artifacts.tokenizer.get("tokenizer")
#         input_ids = tokenizer.encode(src_text, return_tensors="pt")
#         # output = model.generate(input_ids, max_length=50)
#         # output = tokenizer.decode(output[0], skip_special_tokens=True)
#         return input_ids


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchModelArtifact('cls'), TokenizerArtifact("tokenizer"),])
class PytorchLightingService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(),output=JsonOutput(), batch=True)
    def predict(self, parsed_json):
        # test()
        model = self.artifacts.cls
        model.eval()
        # # parsed_json
        tokenizer = self.artifacts.tokenizer.get("tokenizer")
        maxLen=32
        # print(type(parsed_json))
        # print(parsed_json)
        text=parsed_json[0]['text']
        
        # print("text",text)
        # tokenizer = BertTokenizerFast.from_pretrained("/data_200t/chenyaozu/data/base_model/chinese_roberta_L-4_H-512")
        print(dir(tokenizer))
        print(dir(model))
        inputData=tokenizer(text,padding="max_length",max_length=maxLen,return_tensors="pt",truncation=True)
        # # input_tensor = torch.from_numpy(df.to_numpy())
        # # return self.artifacts.model(input).numpy()
        # # return self.artifacts.model(inputData['input_ids'],inputData['token_type_ids'],inputData['attention_mask']).numpy()
        # print(inputData)
        out=model(inputData['input_ids'],inputData['token_type_ids'],inputData['attention_mask'])
        # return inputData['input_ids'].numpy()
        return out[0].tolist()
    
    @bentoml.api(input=JsonInput(),output=JsonOutput(), batch=True)
    def test(self, parsed_json):
        model = self.artifacts.cls
        model.eval()
        # # parsed_json
        tokenizer = self.artifacts.tokenizer.get("tokenizer")
        maxLen=32
        # print(type(parsed_json))
        # print(parsed_json)
        text=parsed_json[0]['text']
        
        # print("text",text)
        # tokenizer = BertTokenizerFast.from_pretrained("/data_200t/chenyaozu/data/base_model/chinese_roberta_L-4_H-512")
        print(dir(tokenizer))
        print(dir(model))
        inputData=tokenizer(text,padding="max_length",max_length=maxLen,return_tensors="pt",truncation=True)
        # # input_tensor = torch.from_numpy(df.to_numpy())
        # # return self.artifacts.model(input).numpy()
        # # return self.artifacts.model(inputData['input_ids'],inputData['token_type_ids'],inputData['attention_mask']).numpy()
        # print(inputData)
        out=model(inputData['input_ids'],inputData['token_type_ids'],inputData['attention_mask'])
        # return inputData['input_ids'].numpy()
        return out[0].tolist()
