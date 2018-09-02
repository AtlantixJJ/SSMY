__author__ = "Jianjin Xu"
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io

def default_weight(shape, name=None):
    # print(shape)
    initial = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.keras.regularizers.L1L2(0,1e-4)
    return tf.get_variable(name=name, shape=shape, initializer=initial, regularizer=regularizer)

def default_conv2d_W(x, kernel, bias=None, stride=1):
    conv = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding="SAME")
    if bias:
        return tf.nn.bias_add(conv, bias)
    return conv

def default_conv2d(x, shape, stride, name=["W","b"]):
    """
    (w,h, inCh, outCh)
    """
    kernel = default_weight(shape, name=name[0])
    bias = default_weight([shape[-1]], name=name[1])
    return default_conv2d_W(x, kernel, bias, stride)