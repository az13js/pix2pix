#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# 从 inputs 和 outputs 里面复制一部分文件到 test 里面的 src 文件夹和 dst 文
# 件夹内。这一步操作是方便在运行 train.py 之前先从训练用的数据上移动一部分出来
# 作为测试集。这将会方便在 tran.py 上使用 --test 选项。

import os
import shutil
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 根据扩展名判断
def isImage(file):
    if os.path.exists(file):
        if os.path.isfile(file):
            fp = file.lower()
            if fp.endswith('.png') or fp.endswith('.jpg') or fp.endswith('.jpeg'):
                return True
    return False

def deleteImages(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        fileList = os.listdir(dir)
        for file in fileList:
            filePath = dir + os.sep + file
            if isImage(filePath):
                os.remove(filePath)
    else:
        print("Error: \"" + dir + "\" is not exists!")

# 读取指定文件夹下的图片文件名，并返回
def getImageFileList(dir):
    result = []
    if os.path.exists(dir) and os.path.isdir(dir):
        fileList = os.listdir(dir)
        for file in fileList:
            filePath = dir + os.sep + file
            if isImage(filePath):
                result.append(filePath)
    return result

# 移动文件数量
moveNum = 4

# 输入文件夹和输出文件夹的位置
inputs = "inputs"
outputs = "outputs"

# 移动目的地
moveSrc = "test" + os.sep + "src"
moveDst = "test" + os.sep + "dst"

# 读取文件列表
inputImages = getImageFileList(inputs)
outputImages = getImageFileList(outputs)

deleteImages(moveSrc)
deleteImages(moveDst)

for i in range(moveNum):
    print("move " + inputImages[i])
    shutil.move(inputImages[i], moveSrc + os.sep + os.path.basename(inputImages[i]))
    print("move " + outputImages[i])
    shutil.move(outputImages[i], moveDst + os.sep + os.path.basename(outputImages[i]))
