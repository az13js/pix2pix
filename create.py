#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
from PIL import Image

def createPix2PixGenerator():
    firstFilter = 64
    inputChannels = 6 # first 3 for input image(handwrite), last 3 for random noise
    outputChannels = 3

    # input-layer: 256 * 256 * 3
    input1 = tf.keras.layers.Input(shape = (256, 256, inputChannels / 2, ))
    input2 = tf.keras.layers.Input(shape = (256, 256, inputChannels / 2, ))
    inputs = tf.keras.layers.concatenate([input1, input2], 3) # Join, double channels

    encoder1Conv = tf.keras.layers.Conv2D(filters=firstFilter, kernel_size=4, strides=2, padding="same") # filters=64
    # 128 * 128 * 64
    encoder1LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder2Conv = tf.keras.layers.Conv2D(filters=firstFilter*2, kernel_size=4, strides=2, padding="same") # filters=128
    # 64 * 64 * 128
    encoder2BatchNorm = tf.keras.layers.BatchNormalization()
    encoder2LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder3Conv = tf.keras.layers.Conv2D(filters=firstFilter*4, kernel_size=4, strides=2, padding="same") # filters=256
    # 32 * 32 * 256
    encoder3BatchNorm = tf.keras.layers.BatchNormalization()
    encoder3LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder4Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 16 * 16 * 512
    encoder4BatchNorm = tf.keras.layers.BatchNormalization()
    encoder4LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder5Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 8 * 8 * 512
    encoder5BatchNorm = tf.keras.layers.BatchNormalization()
    encoder5LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder6Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 4 * 4 * 512
    encoder6BatchNorm = tf.keras.layers.BatchNormalization()
    encoder6LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder7Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 2 * 2 * 512
    encoder7BatchNorm = tf.keras.layers.BatchNormalization()
    encoder7LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder8Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 1 * 1 * 512
    encoder8LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.0) # ReLu: alpha=0.0

    decoder1FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 2 * 2 * 512
    decoder1BatchNorm = tf.keras.layers.BatchNormalization()
    decoder1Dropout = tf.keras.layers.Dropout(rate = 0.5)
    decoder1ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder2FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 4 * 4 * 512
    decoder2BatchNorm = tf.keras.layers.BatchNormalization()
    decoder2Dropout = tf.keras.layers.Dropout(rate = 0.5)
    decoder2ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder3FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 8 * 8 * 512
    decoder3BatchNorm = tf.keras.layers.BatchNormalization()
    decoder3Dropout = tf.keras.layers.Dropout(rate = 0.5)
    decoder3ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder4FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 16 * 16 * 512
    decoder4BatchNorm = tf.keras.layers.BatchNormalization()
    decoder4ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder5FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*4, kernel_size=4, strides=2, padding="same") # filters=256
    # 32 * 32 * 256
    decoder5BatchNorm = tf.keras.layers.BatchNormalization()
    decoder5ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder6FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*2, kernel_size=4, strides=2, padding="same") # filters=128
    # 64 * 64 * 128
    decoder6BatchNorm = tf.keras.layers.BatchNormalization()
    decoder6ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder7FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter, kernel_size=4, strides=2, padding="same") # filters=64
    # 128 * 128 * 64
    decoder7BatchNorm = tf.keras.layers.BatchNormalization()
    decoder7ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    # filters is output channels. Change it if you want to get different outputs
    decoder8FullConv = tf.keras.layers.Conv2DTranspose(filters=outputChannels, kernel_size=4, strides=2, padding="same") # filters=3
    # 256 * 256 * 3
    decoder8Tanh = tf.keras.layers.Activation("tanh")

    e1 = layer = encoder1Conv(inputs)
    layer = encoder1LeakyReLU(layer)

    layer = encoder2Conv(layer)
    e2 = layer = encoder2BatchNorm(layer)
    layer = encoder2LeakyReLU(layer)

    layer = encoder3Conv(layer)
    e3 = layer = encoder3BatchNorm(layer)
    layer = encoder3LeakyReLU(layer)

    layer = encoder4Conv(layer)
    e4 = layer = encoder4BatchNorm(layer)
    layer = encoder4LeakyReLU(layer)

    layer = encoder5Conv(layer)
    e5 = layer = encoder5BatchNorm(layer)
    layer = encoder5LeakyReLU(layer)

    layer = encoder6Conv(layer)
    e6 = layer = encoder6BatchNorm(layer)
    layer = encoder6LeakyReLU(layer)

    layer = encoder7Conv(layer)
    e7 = layer = encoder7BatchNorm(layer)
    layer = encoder7LeakyReLU(layer)

    layer = encoder8Conv(layer)
    layer = encoder8LeakyReLU(layer)

    layer = decoder1FullConv(layer)
    layer = decoder1BatchNorm(layer)
    layer = decoder1Dropout(layer)
    layer = tf.keras.layers.concatenate([layer, e7], 3) # Join, double channels
    layer = decoder1ReLU(layer)

    layer = decoder2FullConv(layer)
    layer = decoder2BatchNorm(layer)
    layer = decoder2Dropout(layer)
    layer = tf.keras.layers.concatenate([layer, e6], 3)
    layer = decoder2ReLU(layer)

    layer = decoder3FullConv(layer)
    layer = decoder3BatchNorm(layer)
    layer = decoder3Dropout(layer)
    layer = tf.keras.layers.concatenate([layer, e5], 3)
    layer = decoder3ReLU(layer)

    layer = decoder4FullConv(layer)
    layer = decoder4BatchNorm(layer)
    layer = tf.keras.layers.concatenate([layer, e4], 3)
    layer = decoder4ReLU(layer)

    layer = decoder5FullConv(layer)
    layer = decoder5BatchNorm(layer)
    layer = tf.keras.layers.concatenate([layer, e3], 3)
    layer = decoder5ReLU(layer)

    layer = decoder6FullConv(layer)
    layer = decoder6BatchNorm(layer)
    layer = tf.keras.layers.concatenate([layer, e2], 3)
    layer = decoder6ReLU(layer)

    layer = decoder7FullConv(layer)
    layer = decoder7BatchNorm(layer)
    layer = tf.keras.layers.concatenate([layer, e1], 3)
    layer = decoder7ReLU(layer)

    layer = decoder8FullConv(layer)
    layer = decoder8Tanh(layer)

    model = tf.keras.models.Model(inputs = [input1, input2], outputs = layer)
    return model

def createPix2PixDiscriminator():
    imageChannels = 3
    firstFilters = 64
    model = tf.keras.Sequential()
    # cGAN input=2
    model.add(tf.keras.layers.Conv2D(input_shape=[256, 256, imageChannels*2], filters=firstFilters, kernel_size=4, strides=2, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    nLarys = 3
    nf_mult = 1
    nf_mult_prev = 1
    for i in range(nLarys):
        nf_mult_prev = nf_mult
        nf_mult = min(2**(i + 1), 8)
        model.add(tf.keras.layers.Conv2D(filters=firstFilters*nf_mult, kernel_size=4, strides=2, padding="same"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # state size: (ndf*M) x N x N
    nf_mult_prev = nf_mult
    nf_mult = min(2**nLarys, 8)
    model.add(tf.keras.layers.Conv2D(filters=firstFilters*nf_mult, kernel_size=4, strides=1, padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # state size: (ndf*M*2) x (N-1) x (N-1)
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same"))
    # state size: 1 x (N-2) x (N-2)

    model.add(tf.keras.layers.Activation("sigmoid"))
    # state size: 1 x (N-2) x (N-2)
    return model

def createMixModel():
    firstFilter = 64
    inputChannels = 6 # first 3 for input image, last 3 for random noise
    outputChannels = 3

    # input-layer: 256 * 256 * 3
    input1 = tf.keras.layers.Input(shape = (256, 256, inputChannels / 2, ))
    input2 = tf.keras.layers.Input(shape = (256, 256, inputChannels / 2, ))
    inputs = tf.keras.layers.concatenate([input1, input2], 3) # Join, double channels

    encoder1Conv = tf.keras.layers.Conv2D(filters=firstFilter, kernel_size=4, strides=2, padding="same") # filters=64
    # 128 * 128 * 64
    encoder1LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder2Conv = tf.keras.layers.Conv2D(filters=firstFilter*2, kernel_size=4, strides=2, padding="same") # filters=128
    # 64 * 64 * 128
    encoder2BatchNorm = tf.keras.layers.BatchNormalization()
    encoder2LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder3Conv = tf.keras.layers.Conv2D(filters=firstFilter*4, kernel_size=4, strides=2, padding="same") # filters=256
    # 32 * 32 * 256
    encoder3BatchNorm = tf.keras.layers.BatchNormalization()
    encoder3LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder4Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 16 * 16 * 512
    encoder4BatchNorm = tf.keras.layers.BatchNormalization()
    encoder4LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder5Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 8 * 8 * 512
    encoder5BatchNorm = tf.keras.layers.BatchNormalization()
    encoder5LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder6Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 4 * 4 * 512
    encoder6BatchNorm = tf.keras.layers.BatchNormalization()
    encoder6LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder7Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 2 * 2 * 512
    encoder7BatchNorm = tf.keras.layers.BatchNormalization()
    encoder7LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.2)

    encoder8Conv = tf.keras.layers.Conv2D(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 1 * 1 * 512
    encoder8LeakyReLU = tf.keras.layers.LeakyReLU(alpha=0.0) # ReLu: alpha=0.0

    decoder1FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 2 * 2 * 512
    decoder1BatchNorm = tf.keras.layers.BatchNormalization()
    decoder1Dropout = tf.keras.layers.Dropout(rate = 0.5)
    decoder1ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder2FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 4 * 4 * 512
    decoder2BatchNorm = tf.keras.layers.BatchNormalization()
    decoder2Dropout = tf.keras.layers.Dropout(rate = 0.5)
    decoder2ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder3FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 8 * 8 * 512
    decoder3BatchNorm = tf.keras.layers.BatchNormalization()
    decoder3Dropout = tf.keras.layers.Dropout(rate = 0.5)
    decoder3ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder4FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*8, kernel_size=4, strides=2, padding="same") # filters=512
    # 16 * 16 * 512
    decoder4BatchNorm = tf.keras.layers.BatchNormalization()
    decoder4ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder5FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*4, kernel_size=4, strides=2, padding="same") # filters=256
    # 32 * 32 * 256
    decoder5BatchNorm = tf.keras.layers.BatchNormalization()
    decoder5ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder6FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter*2, kernel_size=4, strides=2, padding="same") # filters=128
    # 64 * 64 * 128
    decoder6BatchNorm = tf.keras.layers.BatchNormalization()
    decoder6ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    decoder7FullConv = tf.keras.layers.Conv2DTranspose(filters=firstFilter, kernel_size=4, strides=2, padding="same") # filters=64
    # 128 * 128 * 64
    decoder7BatchNorm = tf.keras.layers.BatchNormalization()
    decoder7ReLU = tf.keras.layers.LeakyReLU(alpha=0.0)

    # filters is output channels. Change it if you want to get different outputs
    decoder8FullConv = tf.keras.layers.Conv2DTranspose(filters=outputChannels, kernel_size=4, strides=2, padding="same") # filters=3
    # 256 * 256 * 3
    decoder8Tanh = tf.keras.layers.Activation("tanh")

    e1 = layer = encoder1Conv(inputs)
    layer = encoder1LeakyReLU(layer)

    layer = encoder2Conv(layer)
    e2 = layer = encoder2BatchNorm(layer)
    layer = encoder2LeakyReLU(layer)

    layer = encoder3Conv(layer)
    e3 = layer = encoder3BatchNorm(layer)
    layer = encoder3LeakyReLU(layer)

    layer = encoder4Conv(layer)
    e4 = layer = encoder4BatchNorm(layer)
    layer = encoder4LeakyReLU(layer)

    layer = encoder5Conv(layer)
    e5 = layer = encoder5BatchNorm(layer)
    layer = encoder5LeakyReLU(layer)

    layer = encoder6Conv(layer)
    e6 = layer = encoder6BatchNorm(layer)
    layer = encoder6LeakyReLU(layer)

    layer = encoder7Conv(layer)
    e7 = layer = encoder7BatchNorm(layer)
    layer = encoder7LeakyReLU(layer)

    layer = encoder8Conv(layer)
    layer = encoder8LeakyReLU(layer)

    layer = decoder1FullConv(layer)
    layer = decoder1BatchNorm(layer)
    layer = decoder1Dropout(layer)
    layer = tf.keras.layers.concatenate([layer, e7], 3) # Join, double channels
    layer = decoder1ReLU(layer)

    layer = decoder2FullConv(layer)
    layer = decoder2BatchNorm(layer)
    layer = decoder2Dropout(layer)
    layer = tf.keras.layers.concatenate([layer, e6], 3)
    layer = decoder2ReLU(layer)

    layer = decoder3FullConv(layer)
    layer = decoder3BatchNorm(layer)
    layer = decoder3Dropout(layer)
    layer = tf.keras.layers.concatenate([layer, e5], 3)
    layer = decoder3ReLU(layer)

    layer = decoder4FullConv(layer)
    layer = decoder4BatchNorm(layer)
    layer = tf.keras.layers.concatenate([layer, e4], 3)
    layer = decoder4ReLU(layer)

    layer = decoder5FullConv(layer)
    layer = decoder5BatchNorm(layer)
    layer = tf.keras.layers.concatenate([layer, e3], 3)
    layer = decoder5ReLU(layer)

    layer = decoder6FullConv(layer)
    layer = decoder6BatchNorm(layer)
    layer = tf.keras.layers.concatenate([layer, e2], 3)
    layer = decoder6ReLU(layer)

    layer = decoder7FullConv(layer)
    layer = decoder7BatchNorm(layer)
    layer = tf.keras.layers.concatenate([layer, e1], 3)
    layer = decoder7ReLU(layer)

    layer = decoder8FullConv(layer)
    layer = decoder8Tanh(layer)

    imageChannels = 3
    firstFiltersD = 64

    # Discriminator's input is 6 channels, first 3 for handwrite, last 3 for realpicture
    # In this case, Generator greate a image like realpicture.
    inputLayerForDiscriminator = tf.keras.layers.concatenate([input1, layer], 3)
    # cGAN input=2
    dl = tf.keras.layers.Conv2D(input_shape=[256, 256, imageChannels*2], filters=firstFiltersD, kernel_size=4, strides=2, padding="same", trainable=False)
    d = dl(inputLayerForDiscriminator)
    dl = tf.keras.layers.LeakyReLU(alpha=0.2, trainable=False)
    d = dl(d)

    nLarys = 3
    nf_mult = 1
    nf_mult_prev = 1
    for i in range(nLarys):
        nf_mult_prev = nf_mult
        nf_mult = min(2**(i + 1), 8)
        dl = tf.keras.layers.Conv2D(filters=firstFiltersD*nf_mult, kernel_size=4, strides=2, padding="same", trainable=False)
        d = dl(d)
        dl = tf.keras.layers.BatchNormalization(trainable=False)
        d = dl(d)
        dl = tf.keras.layers.LeakyReLU(alpha=0.2, trainable=False)
        d = dl(d)

    # state size: (ndf*M) x N x N
    nf_mult_prev = nf_mult
    nf_mult = min(2**nLarys, 8)
    dl = tf.keras.layers.Conv2D(filters=firstFiltersD*nf_mult, kernel_size=4, strides=1, padding="same", trainable=False)
    d = dl(d)
    dl = tf.keras.layers.BatchNormalization(trainable=False)
    d = dl(d)
    dl = tf.keras.layers.LeakyReLU(alpha=0.2, trainable=False)
    d = dl(d)

    # state size: (ndf*M*2) x (N-1) x (N-1)
    dl = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1, padding="same", trainable=False)
    d = dl(d)
    # state size: 1 x (N-2) x (N-2)

    dl = tf.keras.layers.Activation("sigmoid", trainable=False)
    d = dl(d)
    model = tf.keras.models.Model(inputs = [input1, input2], outputs = d)
    return model

print('Generator')
n = createPix2PixGenerator()

print('Save')
tf.keras.models.save_model(
    n,
    'Pix2PixGenerator.tf.keras.model',
    overwrite=True,
    include_optimizer=False
)

print('Discriminator')
n = createPix2PixDiscriminator()

print('Save')
tf.keras.models.save_model(
    n,
    'Pix2PixDiscriminator.tf.keras.model',
    overwrite=True,
    include_optimizer=False
)

print('Mix')
n = createMixModel()

print('Save')
tf.keras.models.save_model(
    n,
    'Pix2PixMix.tf.keras.model',
    overwrite=True,
    include_optimizer=False
)
