# sentence Bert api部署示例

# 保存模型

> python save.py

# 运行测试

> bentoml run PytorchLightingService:20211204231545_88CDD2 predict --input '{"text":"你好"}'

> bentoml serve SentenceBertService:20211205012215_447D1E


> {"textA":"ewweew","textB":"ewweew"}

# 推送到服务器

> bentoml push TokenizerService:20211204223058_B31A26 --yatai-url=127.0.0.1:50051
# Docker build过程中使用代理

https://terrychan.org/2021/12/docker-build%e8%bf%87%e7%a8%8b%e4%b8%ad%e4%bd%bf%e7%94%a8%e4%bb%a3%e7%90%86/
# 更多示例

https://docs.bentoml.org/en/latest/concepts.html?highlight=docker#api-server-dockerization

# Find the local path of the latest version IrisClassifier saved bundle

saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)

# Build docker image using saved_path directory as the build context, replace the

# {username} below to your docker hub account name

docker build -t {username}/iris_classifier_bento_service $saved_path

# Run a container with the docker image built and expose port 5000

docker run -p 5000:5000 {username}/iris_classifier_bento_service

# Push the docker image to docker hub for deployment

docker push {username}/iris_classifier_bento_service

# Find the local path of the latest version IrisClassifier saved bundle

saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)

# Replace {docker_username} with your Docker Hub username

docker build -t {docker_username}/iris-classifier $saved_path docker push {docker_username}/iris-classifier

https://docs.bentoml.org/en/latest/quickstart.html

https://colab.research.google.com/github/bentoml/gallery/blob/master/scikit-learn/iris-classifier/iris-classifier.ipynb#scrollTo=mgakbz1JxMpc

```python

% % writefile
iris_classifier.py
from bentoml import env, artifacts, api, BentoService, web_static_content
from bentoml.adapters import DataframeInput
from bentoml.artifact import SklearnModelArtifact


@env(auto_pip_dependencies=True)
@artifacts([SklearnModelArtifact('model')])
@web_static_content('./static')
class IrisClassifier(BentoService):

    @api(input=DataframeInput(), batch=True)
    def test(self, df):
        # Optional pre-processing, post-processing code goes here
        return self.artifacts.model.predict(df)





```

```commandline
 2009  saved_path=$(bentoml get SentenceBertService:20211205130816_BE4579 --print-location --quiet)
 2010  docker build -t napoler/sim_service $saved_path
 2011  docker build -t napoler/sim_service $saved_path http_proxy=http://127.0.0.1:38573
 2013  docker build -t napoler/sim_service $saved_path --build-arg  http_proxy=http://127.0.0.1:38573
 2015  docker run -p 5002:5000 napoler/sim_service:latest


```

