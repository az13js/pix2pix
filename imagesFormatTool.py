#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# 将当前目录下的src内的图片文件整理成256x256px的图片，放入文件夹outputs内。
# 然后重命名outputs文件夹下的文件以随机打乱文件。
# 最后将outputs文件夹下的文件转换成黑白色放入inputs。

import os
import numpy
from PIL import Image
import random
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

# 此函数功能与fileProcess相反
def numpyToFile(data, file):
    imageObject = Image.fromarray(numpy.uint8((numpy.clip(data, -0.5, 0.5) + 0.5) * 255.0))
    imageObject.save(file)

# dir C:\example 或 \usr\example 不需要在结尾加分割符
# 一般地，训练神经网络需要将图片重新调整大小，可以定义width, high参数
def getImagesNumpyFrom(dir, width = 0, high = 0):
    result = []
    if os.path.exists(dir):
        fileList = os.listdir(dir)
        for file in fileList:
            filePath = dir + os.sep + file # 文件的全路径
            if os.path.exists(filePath):
                if os.path.isfile(filePath) and isImage(filePath):
                    # 这里读取filePath
                    result.append(fileToNumpy(filePath, width, high))
    return result

# 将src的文件读取出来，按照宽度高度居中对齐，然后按照顺序保存到dst下
# 例如 formateImagesFrom("example", "dstExample", 256, 256)
# 返回值是转换的文件的数量
def formateImagesFrom(src, dst, width = 0, high = 0):
    idx = 0
    if os.path.exists(src):
        fileList = os.listdir(src)
        for file in fileList:
            filePath = src + os.sep + file # 文件的全路径
            if os.path.exists(filePath):
                if os.path.isfile(filePath) and isImage(filePath):
                    # 这里读取filePath
                    data = fileToNumpy(filePath, width, high)
                    idx = idx + 1
                    numpyToFile(data, dst  + os.sep + str(idx) + '.png')
    return idx

# 将src的文件读取出来，转换成黑白色保存到dst下
# 例如 formateImagesRGBToLFrom("example", "dstExample")
def formateImagesRGBToLFrom(src, dst):
    if os.path.exists(src):
        fileList = os.listdir(src)
        for file in fileList:
            filePath = src + os.sep + file # 文件的全路径
            if os.path.exists(filePath):
                if os.path.isfile(filePath) and isImage(filePath):
                    # 这里读取filePath
                    imageObject = Image.open(filePath).convert('L')
                    imageObject.load()
                    imageObject.save(dst + os.sep + file)

# 重命名，打乱图片
def shuffFile(dir, start, end, ext = ".png"):
    for i in range(start, end):
        swap = random.randint(i, end)
        if (swap != i):
            os.rename(dir + os.sep + str(swap) + ext, dir + os.sep + "temp" + ext)
            os.rename(dir + os.sep + str(i) + ext, dir + os.sep + str(swap) + ext)
            os.rename(dir + os.sep + "temp" + ext, dir + os.sep + str(i) + ext)

deleteImages("outputs")
deleteImages("inputs")
num = formateImagesFrom("src", "outputs", 256, 256)
shuffFile("outputs", 1, num)
formateImagesRGBToLFrom("outputs", "inputs")
