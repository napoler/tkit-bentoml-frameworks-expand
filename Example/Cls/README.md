# 运行测试


#保存模型
python tokenizer_save.py 
#运行测试
bentoml run TokenizerService:20211204223058_B31A26 predict --input '{"text":"你好"}'
#推送到服务器
bentoml push TokenizerService:20211204223058_B31A26 --yatai-url=127.0.0.1:50051
