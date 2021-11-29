
#encoding=utf-8
from __future__ import absolute_import, unicode_literals
from os import symlink
import sys

 
# 切换到上级目录
sys.path.append("../")
# 引入本地库
import Demo

Demo =Demo.Demo()
Demo.fun()

os.add("a.txt")