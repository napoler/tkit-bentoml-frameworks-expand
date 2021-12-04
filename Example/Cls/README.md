# 运行测试

# 保存模型

python cls_save.py

# 运行测试

bentoml run PytorchLightingService:20211204231545_88CDD2 predict --input '{"text":"你好"}'

# 推送到服务器

bentoml push TokenizerService:20211204223058_B31A26 --yatai-url=127.0.0.1:50051
