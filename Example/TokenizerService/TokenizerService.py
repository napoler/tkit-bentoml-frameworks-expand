import sys

import bentoml
from bentoml.adapters import JsonInput

# 移动到上级目录
sys.path.append("../../")

from tkit

-bentoml - frameworks - expand.TokenizerArtifact
import TokenizerArtifact

"""
保存BertTokenizer分词方案

"""


@bentoml.env(pip_packages=["transformers>=4.12.5"])
@bentoml.artifacts([TokenizerArtifact("tokenizer")])
class TokenizerService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), batch=False)
    def predict(self, parsed_json):
        src_text = parsed_json.get("text")
        # model = self.artifacts.gptModel.get("model")
        tokenizer = self.artifacts.tokenizer.get("tokenizer")
        input_ids = tokenizer.encode(src_text, return_tensors="pt")
        # output = model.generate(input_ids, max_length=50)
        # output = tokenizer.decode(output[0], skip_special_tokens=True)
        return input_ids
