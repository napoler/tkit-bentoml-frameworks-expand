from transformers import AutoTokenizer

from TokenizerService import TokenizerService

ts = TokenizerService()
model_name = "uer/chinese_roberta_L-2_H-128"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Option 1: Pack using dictionary (recommended)
artifact = {"tokenizer": tokenizer}
ts.pack("tokenizer", artifact)
# Option 2: pack using the name of the model
# ts.pack("gptModel","gpt2")
# Note that while packing using the name of the model,
# ensure that the model can be loaded using
# transformers.AutoModelWithLMHead (eg GPT, Bert, Roberta etc.)
# If this is not the case (eg AutoModelForQuestionAnswering, BartModel etc)
# then pack the model by passing a dictionary
# with the model and tokenizer declared explicitly
saved_path = ts.save()


"""
# 运行测试
#保存模型
python tokenizer_save.py 
#运行测试
bentoml run TokenizerService:20211204223058_B31A26 predict --input '{"text":"你好"}'
#推送到服务器
bentoml push TokenizerService:20211204223058_B31A26 --yatai-url=127.0.0.1:50051

"""