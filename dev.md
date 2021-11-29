
修改

- setup.py
- config.py
- docs_tools/autodocs.sh  # 需要修改sphinx-apidoc -o ./source/res ../Demo这句 用来修改成扫描的目录
- config.py

# 自动发布pypi


获取 pypitonken

https://pypi.org/manage/account/

添加

https://github.com/napoler/python_libs_demo/settings/secrets/actions

> PIPY 


修改

> .github/workflows/upload-to-pip.yml









自动执行上传

> bash docs_tools/autodocs.sh

## 快速上传操作
可以自动查找依赖，然后上传
```
sh upload.sh
```
详细

```
# 安装依赖查找库
pip install pipreqs
pip install twine
#遇到已经存在 强制覆盖 requirements.txt
pipreqs ./ --force

rm -rf dist
#打包
python3 setup.py sdist
python setup.py bdist_wheel --universal # 打包为无需build的wheel。其中--universal表示py2和py3通用的pure python模块。不满足通用或pure条件的模块不需加此参数
#python3 setup.py install
#上传
# python3 setup.py sdist upload
twine upload dist/*


```




更多开发说明参考这里 https://python-packaging-zh.readthedocs.io/zh_CN/latest/minimal.html


## 如何向PyPi(pip)提交模块

https://www.notion.so/terrychanorg/PyPi-pip-b371898f30ec4f268688edebab8d7ba1

## 提交到anaconda

https://docs.anaconda.com/anacondaorg/user-guide/tasks/work-with-packages/


##  MANIFEST.in 文件

 MANIFEST.in 文件，文件内容就是需要包含在分发包中的文件。一个 MANIFEST.in 文件如下：

```
include *.txt
recursive-include examples *.txt *.py
prune examples/sample?/build
```

MANIFEST.in 文件的编写规则可参考：https://docs.python.org/3.6/distutils/sourcedist.html
