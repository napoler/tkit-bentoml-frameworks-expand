# 运行测试

# 保存模型

python sk_save.py

# 运行测试

bentoml run TokenizerService:20211204223058_B31A26 predict --input '{"text":"你好"}'

# 推送到服务器

bentoml push TokenizerService:20211204223058_B31A26 --yatai-url=127.0.0.1:50051

# 文档

http://0.0.0.0:5000/docs

Define and Debug Services Services are the core components of BentoML where the serving logic is defined. With the model
saved in the model store, we can define the service by creating a Python file bento.py in the working directory with the
following contents. In the example below, we defined numpy.ndarray as the input and output type. More options like
pandas.dataframe and PIL.image are also supported IO types, see @API and IO Descriptors.

# bento.py

import bentoml import bentoml.sklearn import numpy as np

from bentoml.io import NumpyNdarray

# Load the runner for the latest ScikitLearn model we just saved

iris_clf_runner = bentoml.sklearn.load_runner("iris_classifier_model:latest")

# Create the iris_classifier_service with the ScikitLearn runner

svc = bentoml.Service("iris_classifier_service", runners=[iris_clf_runner])

# Create API function with pre- and post- processing logic

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_ndarray: np.ndarray) -> np.ndarray:
# Define pre-processing logic result = iris_clf_runner.run(input_ndarray)

# Define post-processing logic

return result

