import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from resnet import *

def Generator(t_image, model, reuse=False):
    '''
    生成器
    使用resnet-50

    :param t_image:
    :param model:
    :param reuse:
    :return:
    '''

    with tf.variable_scope("Generator", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name='Generator/in')
        n = model(n)
        return n

def Discriminator(input_images, model, reuse=False):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("Discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(input_images, name='input/images')
        n = model(n)
        n = FlattenLayer(n, name='Discriminator/flatten')
        n = DenseLayer(n, n_units=1, act=tf.identity,
                W_init = w_init, name='Discriminator/dense')
        logits = n.outputs
        # Wasserstein GAN doesn't need the sigmoid output
        # net_ho.outputs = tf.nn.sigmoid(net_ho.outputs)

    return n, logits


