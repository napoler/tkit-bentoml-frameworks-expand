# 运行测试

# 保存模型

> python cls_save.py

# 运行测试

> bentoml run PytorchLightingService:20211204231545_88CDD2 predict --input '{"text":"你好"}'

> bentoml serve PytorchLightingService:20211204231829_3EC4D6

# 推送到服务器

> bentoml push TokenizerService:20211204223058_B31A26 --yatai-url=127.0.0.1:50051

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
