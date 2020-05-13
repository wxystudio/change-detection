#!/usr/bin/env python
# coding: utf-8

"""
@File     : water_hole
@Copyright: INNNO Co.Ltd
@author   : xiaoyu wang
@Date     : 2019/11/12
@Desc     :
"""
import cv2
import numpy as np
from keras import backend as K
from keras.models import Input, Model
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, DepthwiseConv2D, UpSampling2D, \
    BatchNormalization, Activation, concatenate, add, Conv2DTranspose

__all__ = ["get_model"]

input_shape = (512, 512, 3)


def pre_relu6(x):
    return K.relu(x, alpha=0.001, max_value=6)


def bn_conv2D(x, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), use_bias=False):
    x = Conv2D(filters, kernel_size, padding='same', strides=strides,
               kernel_initializer='he_normal', dilation_rate=dilation_rate, use_bias=use_bias)(x)
    x = BatchNormalization(axis=-1, scale=False, epsilon=1e-3)(x)
    return Activation(pre_relu6)(x)


def bn_dw_conv2D(x, kernel_size, strides=(1, 1), dilation_rate=(1, 1), use_bias=False):
    x = DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=1, padding='same', use_bias=use_bias,
                        dilation_rate=dilation_rate, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=-1, scale=False, epsilon=1e-3)(x)
    return Activation(pre_relu6)(x)


def bn_relu6(x):
    x = BatchNormalization(axis=-1, scale=False, epsilon=1e-3)(x)
    x = Activation(pre_relu6)(x)
    return x


def down_sampling_block(x, filters):
    in_shape = K.int_shape(x)[-1]
    if in_shape < filters:
        n_filters = filters - in_shape
    else:
        n_filters = filters

    x1 = Conv2D(n_filters, (3, 3), strides=(2, 2), padding="same", use_bias=False)(x)
    if in_shape < filters:
        max_pool = MaxPooling2D(pool_size=(2, 2))(x)
        x = concatenate([x1, max_pool], axis=-1)
    else:
        x = x1

    return bn_relu6(x)


def dab_model(x, d, k_size):
    in_shape = K.int_shape(x)[-1]
    x1 = bn_conv2D(x, in_shape // 2, (k_size, k_size), strides=(1, 1))

    x2 = bn_dw_conv2D(x1, (k_size, 1), strides=(1, 1))
    x2 = bn_dw_conv2D(x2, (1, k_size), strides=(1, 1))

    x3 = bn_dw_conv2D(x1, (k_size, 1), strides=(1, 1), dilation_rate=(d, 1))
    x3 = bn_dw_conv2D(x3, (1, k_size), strides=(1, 1), dilation_rate=(1, d))

    x2 = bn_relu6(add([x2, x3]))
    x2 = Conv2D(in_shape, (1, 1), padding="same", strides=(1, 1), kernel_initializer='he_normal', use_bias=True)(x2)
    return bn_relu6(add([x, x2]))


def get_model():
    inputs = Input(input_shape)
    down1 = AveragePooling2D(pool_size=(2, 2))(inputs)
    down2 = AveragePooling2D(pool_size=(2, 2))(down1)
    down3 = AveragePooling2D(pool_size=(2, 2))(down2)

    x = bn_conv2D(inputs, 32, (3, 3), strides=(2, 2))
    x = bn_conv2D(x, 32, (3, 3), strides=(1, 1))
    x = bn_conv2D(x, 32, (3, 3), strides=(1, 1))

    x = bn_relu6(concatenate([x, down1], axis=-1))

    x1 = down_sampling_block(x, 64)
    d1 = [1, 2, 5]
    block1 = len(d1)
    for i, d in zip(range(0, block1), d1):
        if i == 0:
            x = dab_model(x1, d, k_size=3)
        else:
            x = dab_model(x, d, k_size=3)

    x = bn_relu6(concatenate([x, x1, down2], axis=-1))

    x2 = down_sampling_block(x, 128)
    d2 = [1, 2, 3, 5, 3, 5]
    block2 = len(d2)
    for i, d in zip(range(0, block2), d2):
        if i == 0:
            x = dab_model(x2, d, k_size=3)
        else:
            x = dab_model(x, d, k_size=3)
    x = bn_relu6(concatenate([x, x2, down3], axis=-1))

    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', dilation_rate=(2, 2))(x)
    x = bn_conv2D(concatenate([x, down2], axis=-1), 32, (3, 3), strides=(1, 1))
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', dilation_rate=(2, 2))(x)
    x = bn_conv2D(concatenate([x, down1], axis=-1), 16, (3, 3), strides=(1, 1))
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(x)
    outputs = UpSampling2D(size=(2, 2))(outputs)
    return Model(inputs=[inputs], outputs=[outputs])


def img_prepare(img):
    x = cv2.resize(img, (input_shape[1], input_shape[0]))
    x.astype(np.float32)
    return x / 255.0 - 0.5


def predict(model, img, th=0.5):
    h, w = img.shape[:2]
    x = img_prepare(img)
    pmask = model.predict_on_batch(np.array([x]))
    mask = (pmask[0])[:, :, 0]
    mask[mask < th] = 0
    mask[mask >= th] = 1
    mask.astype(np.uint8)
    return cv2.resize(mask, (w, h), cv2.INTER_NEAREST)


if __name__ == '__main__':
    from keras.utils.vis_utils import plot_model

    input_shape = (512, 512, 3)
    model = get_model(input_shape)
    model.summary()
    plot_model(model, "vnet.svg", show_shapes=True)
    model.save_weights("vnet.h5")
    pass
