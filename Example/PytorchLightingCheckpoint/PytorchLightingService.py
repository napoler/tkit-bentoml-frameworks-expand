# from model.SequenceClassificationClear import SequenceClassification
# 移动到上级目录
import sys

sys.path.append("../../")
import bentoml
from tkitBentomlFrameworksExpand.TokenizerArtifact import TokenizerArtifact
from bentoml.adapters import JsonInput, JsonOutput
# from bentoml.adapters import DataframeInput
from bentoml.frameworks.pytorch import PytorchModelArtifact


# add cache
# from functools import lru_cache


@bentoml.env(infer_pip_packages=True)
# @bentoml.artifacts([PytorchModelArtifact('cls'), TokenizerArtifact("tokenizer"),])
@bentoml.artifacts([PytorchModelArtifact('cls'), TokenizerArtifact("tokenizer"), ])
class PytorchLightingService(bentoml.BentoService):
    # @lru_cache()
    @bentoml.api(input=JsonInput(), output=JsonOutput(), batch=True)
    def predict(self, parsed_json):
        # test()
        model = self.artifacts.cls
        model.eval()
        # # parsed_json
        tokenizer = self.artifacts.tokenizer.get("tokenizer")
        maxLen = 32
        # print(type(parsed_json))
        # print(parsed_json)
        text = parsed_json[0]['text']

        # print("text",text)

        # print(dir(tokenizer))
        # print(dir(model))
        inputData = tokenizer(text, padding="max_length", max_length=maxLen, return_tensors="pt", truncation=True)
        # # input_tensor = torch.from_numpy(df.to_numpy())
        # # return self.artifacts.model(input).numpy()
        # # return self.artifacts.model(inputData['input_ids'],inputData['token_type_ids'],inputData['attention_mask']).numpy()
        # print(inputData)
        out = model(inputData['input_ids'], inputData['attention_mask'])
        # return inputData['input_ids'].numpy()
        # outjson={"data":out[0].tolist()}
        return out.tolist()

    @bentoml.api(input=JsonInput(), output=JsonOutput(), batch=True)
    def test(self, parsed_json):
        model = self.artifacts.cls
        model.eval()
        # # parsed_json
        tokenizer = self.artifacts.tokenizer.get("tokenizer")
        maxLen = 32
        # print(type(parsed_json))
        # print(parsed_json)
        text = parsed_json[0]['text']

        # print("text",text)
        # tokenizer = BertTokenizerFast.from_pretrained("/data_200t/chenyaozu/data/base_model/chinese_roberta_L-4_H-512")
        print(dir(tokenizer))
        print(dir(model))
        inputData = tokenizer(text, padding="max_length", max_length=maxLen, return_tensors="pt", truncation=True)
        # # input_tensor = torch.from_numpy(df.to_numpy())
        # # return self.artifacts.model(input).numpy()
        # # return self.artifacts.model(inputData['input_ids'],inputData['token_type_ids'],inputData['attention_mask']).numpy()
        # print(inputData)
        out = model(inputData['input_ids'], inputData['token_type_ids'], inputData['attention_mask'])
        # return inputData['input_ids'].numpy()
        return out[0].tolist()
