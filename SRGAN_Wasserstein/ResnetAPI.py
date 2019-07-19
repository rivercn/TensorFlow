import tensorflow as tf
import  collections
import  time
from  datetime import datetime
import math
import  tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import nets
import pre

slim = tf.contrib.slim
class Model(object):
    def __init__(self,
                 num_class,
                 is_training,
                 fixed_resize_side,
                 default_image_size):
        self._num_classes = num_class
        self._is_training = is_training
        self._fixed_resize_side = fixed_resize_side
        self._default_image_size = default_image_size

    def num_classed(self):
        return self._num_classes

    def preprocess(self,inputs):
        #预处理
        preprocessed_inputs = preprocessing.