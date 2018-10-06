#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os
import random
import tensorflow as tf
from PIL import Image
import numpy

def getParam(param):
    for p in sys.argv:
        sub = p.split("=")
        if len(sub) > 0:
            if sub[0] == param:
                return p.replace(param + "=", "")
    return ""

def isImage(file):
    if os.path.exists(file):
        if os.path.isfile(file):
            fp = file.lower()
            if fp.endswith('.png') or fp.endswith('.jpg') or fp.endswith('.jpeg'):
                return True
    return False

def getImageFileList(dir):
    result = []
    if os.path.exists(dir) and os.path.isdir(dir):
        fileList = os.listdir(dir)
        for file in fileList:
            filePath = dir + os.sep + file
            if isImage(filePath):
                result.append(filePath)
    return result

def getInputsOutputs():
    inputs = getParam("--inputs")
    outputs = getParam("--outputs")

    if "" == inputs:
        inputs = "predict" + os.sep + "src"
    if "" == outputs:
        outputs = "predict" + os.sep + "dst"
    return inputs, outputs

# 利用图像库读取图片并转换成张量，然后返回
# 返回的图像是RGB三通道的，形状依次是：宽、高、通道
# 返回值的范围是-0.5到0.5之间，可能等于-0.5，也可能等于0.5
# width和high参数可以用居中的方式去缩放图片
def fileToNumpy(file, width = 0, high = 0):
    imageObject = Image.open(file).convert('RGB')
    if width > 0 and high > 0:
        if (width != imageObject.size[0] or high != imageObject.size[1]):
            # 计算宽度和高度各自需要的放大比例
            walpha = width / imageObject.size[0]
            halpha = high / imageObject.size[1]
            minAlpha = min(walpha, halpha)
            # 计算出图片缩放后不包括边框的宽度和高度
            wImage = int(minAlpha * imageObject.size[0])
            hImage = int(minAlpha * imageObject.size[1])
            # 计算出最终包括边框的图片里，图片内容的左上角位置
            x = int((width - wImage) / 2)
            y = int((high - hImage) / 2)
            # 调用PIL方法处理图片
            imageObject = imageObject.resize((wImage, hImage), Image.ANTIALIAS)
            temp = Image.new('RGB', (width, high), color=(0, 0, 0))
            temp.paste(imageObject, (x, y))
            imageObject = temp
    return numpy.asarray(imageObject, dtype = numpy.float16) / 255.0 - 0.5

# 此函数功能与fileToNumpy相反
def numpyToFile(data, file):
    imageObject = Image.fromarray(numpy.uint8((numpy.clip(data, -0.5, 0.5) + 0.5) * 255.0))
    imageObject.save(file)

def createGeneratorInputs(files):
    input1 = []
    input2 = []
    for file in files:
        if isImage(file):
            input1.append(fileToNumpy(file, 256, 256))
            input2.append(numpy.random.rand(256, 256, 3) - 0.5)
    return [numpy.array(input1), numpy.array(input2)]

def deleteImages(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        fileList = os.listdir(dir)
        for file in fileList:
            filePath = dir + os.sep + file
            if isImage(filePath):
                os.remove(filePath)
    else:
        print("Error: \"" + dir + "\" is not exists!")

def runGenerator(generator, files, output):
    outputs = generator.predict(createGeneratorInputs(files))
    outputFiles = []
    name = 1
    for imageMatrix in outputs:
        while os.path.exists(output + os.sep + str(name) + ".png") and os.path.isfile(output + os.sep + str(name) + ".png"):
            name = name + 1
        numpyToFile(imageMatrix, output + os.sep + str(name) + ".png")
        outputFiles.append(output + os.sep + str(name) + ".png")
        name = name + 1
    return outputFiles

# 参数
(inputs, outputs) = getInputsOutputs()
print("inputs =", inputs)
print("outputs =", outputs)

# 文件列表
inputImages = getImageFileList(inputs)
print("Inputs number :", len(inputImages))

# 加载模型
print("Loading Pix2PixGenerator.tf.keras.model")
generator = tf.keras.models.load_model('Pix2PixGenerator.tf.keras.model', compile=False)

# 预测
results = generator.predict(createGeneratorInputs(inputImages))

# 保存
deleteImages(outputs)
name = 0
for data in results:
    name = name + 1
    numpyToFile(data, outputs + os.sep + str(name) + ".png")
