#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os
import random
import tensorflow as tf
from PIL import Image
import numpy

def shuffle(a, b):
    if len(a) != len(b):
        return False

    size = len(a)
    if size == 1:
        return a, b

    for i in range(size - 1):
        select = random.randint(i, size - 1)
        if i < select:
            temp = a[i]
            a[i] = a[select]
            a[select] = temp
            temp = b[i]
            b[i] = b[select]
            b[select] = temp
    return a, b

#(a, b) = (["a", "b", "c", "d"], ["A", "B", "C", "D"])
#shuffle(a, b)
#print(a, b)

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
        inputs = "inputs"
    if "" == outputs:
        outputs = "outputs"
    return inputs, outputs

def getTrainParams():
    batch = getParam("--batch")
    bcount = getParam("--bcount")
    count = getParam("--count")

    if "" == batch:
        batch = 2
    else:
        batch = int(batch)
    if "" == bcount:
        bcount = 1
    else:
        bcount = int(bcount)
    if "" == count:
        count = 1
    else:
        count = int(count)
    return batch, bcount, count

def addGD2Mix(generator, discriminator, mix):
    # 1, copy weights from generator's layers, save in a list, only length > 0 we save
    generatorWeights = []
    for layer in generator.layers:
        if "trainable" in layer.get_config().keys():
           weight = layer.get_weights()
           if len(weight) > 0:
               generatorWeights.append(weight)

    # 2, copy weights from discriminator's layers, save in a list, only length > 0 we save
    discriminatorWeights = []
    for layer in discriminator.layers:
       if "trainable" in layer.get_config().keys():
           weight = layer.get_weights()
           if len(weight) > 0:
               discriminatorWeights.append(weight)

    # 3, set weights from generatorWeights to mix's generator-layers
    index = 0
    for layer in mix.layers:
        config = layer.get_config()
        if "trainable" in config.keys():
            if True == config["trainable"]:
                weight = layer.get_weights()
                if len(weight) > 0:
                    layer.set_weights(generatorWeights[index])
                    index = index + 1

    # 4, set weights from discriminatorWeights to mix's discriminator-layers
    index = 0
    for layer in mix.layers:
        config = layer.get_config()
        if "trainable" in config.keys():
           if False == config["trainable"]:
               weight = layer.get_weights()
               if len(weight) > 0:
                   layer.set_weights(discriminatorWeights[index])
                   index = index + 1

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

def trainDiscriminator(discriminator, count, trust, fake):
    data = []
    label = []
    k = -1
    for file in trust[0]:
        k = k + 1
        data.append(numpy.concatenate((fileToNumpy(trust[0][k], 256, 256), fileToNumpy(trust[1][k], 256, 256)), axis = 2))
        label.append(numpy.ones((16, 16, 1)))
        data.append(numpy.concatenate((fileToNumpy(fake[0][k], 256, 256), fileToNumpy(fake[1][k], 256, 256)), axis = 2))
        label.append(numpy.zeros((16, 16, 1)))
    shuffle(data, label)
    return discriminator.fit(numpy.array(data), numpy.array(label), len(label), count)

def trainMix(mix, count, inputs):
    label = []
    for file in inputs:
        label.append(numpy.ones((16, 16, 1)))
    return mix.fit(createGeneratorInputs(inputs), numpy.array(label), len(label), count)

def separateMix(mix, generator):
    generatorWeights = []
    for layer in mix.layers:
        config = layer.get_config()
        if "trainable" in config.keys():
            if True == config["trainable"]:
                weight = layer.get_weights()
                if len(weight) > 0:
                    generatorWeights.append(weight)
    index = 0
    for layer in generator.layers:
        if "trainable" in layer.get_config().keys():
           weight = layer.get_weights()
           if len(weight) > 0:
               layer.set_weights(generatorWeights[index])
               index = index + 1

def trainGenerator(generator, count, inputs, outputs):
    data = createGeneratorInputs(inputs)
    label = []
    for file in outputs:
        label.append(fileToNumpy(file, 256, 256))
    return generator.fit(data, numpy.array(label), len(label), count)

# 参数
(inputs, outputs) = getInputsOutputs()
print("inputs =", inputs)
print("outputs =", outputs)

(batch, bcount, count) = getTrainParams()
print("batch =" , batch)
print("bcount =" , bcount)
print("count =" , count)

# 文件列表
inputImages = getImageFileList(inputs)
outputImages = getImageFileList(outputs)
if len(inputImages) != len(outputImages):
    print("inputImages != outputImages")
    exit()
shuffle(inputImages, outputImages)
print("Inputs-outputs number :", len(inputImages))

# 加载模型
print("Loading Pix2PixGenerator.tf.keras.model,Pix2PixDiscriminator.tf.keras.model,Pix2PixMix.tf.keras.model")
generator = tf.keras.models.load_model('Pix2PixGenerator.tf.keras.model', compile=False)
discriminator = tf.keras.models.load_model('Pix2PixDiscriminator.tf.keras.model', compile=False)
mix = tf.keras.models.load_model('Pix2PixMix.tf.keras.model', compile=False)

# 配置优化算法
print("Compile")
generator.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999), loss = "MSE")
discriminator.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999), loss = "MSE")
mix.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5, beta_2=0.999), loss = "MSE")

# 训练
print("Train")
for countidx in range(count):
    print("count :", countidx + 1)
    for bcidx in range(bcount):
        print("batch count :", bcidx + 1)
        start = 0
        end = start + batch
        while start < len(inputImages):
            deleteImages("fake")
            outputFileList = runGenerator(generator, inputImages[start : end], "fake")
            print("Train discriminator")
            trainDiscriminator(discriminator, bcount, [inputImages[start : end], outputImages[start : end]], [inputImages[start : end], outputFileList])
            addGD2Mix(generator, discriminator, mix)
            print("Train mix")
            trainMix(mix, bcount, inputImages[start : end])
            separateMix(mix, generator)
            print("Train generator")
            trainGenerator(generator, bcount, inputImages[start : end], outputImages[start : end])
            print("Save generator and discriminator")
            tf.keras.models.save_model(generator, "Pix2PixGenerator.tf.keras.model", include_optimizer=False)
            tf.keras.models.save_model(discriminator, "Pix2PixDiscriminator.tf.keras.model", include_optimizer=False)
            print("Ok")
            start = end
            end = start + batch