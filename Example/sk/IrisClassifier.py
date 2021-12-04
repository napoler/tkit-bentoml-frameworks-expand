# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""
import bentoml
from bentoml import web_static_content
from bentoml.adapters import DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

# @save_to_dir("test.py") # 测试保存文件
@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([SklearnModelArtifact('model')])
@web_static_content('./static')
class IrisClassifier(bentoml.BentoService):
    """
    IrisClassifier分类接口

    测试说明


    """

    @bentoml.api(input=DataframeInput(), batch=True)
    def predict(self, df, metadata={"aa": 2}):
        """
        预测接口
        :param df:
        :return:
        """
        return self.artifacts.model.predict(df)


if __name__ == '__main__':
    pass
