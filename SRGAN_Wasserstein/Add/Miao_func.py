import tensorflow as tf
import numpy as  np
from tensorflow.python.framework import ops





def Miao(feature):
    P = tf.maximum(feature, 0)
    N = tf.minimum(feature, 0)
    P = tf.multiply(P, -1)
    P = tf.exp(P)
    P = tf.multiply(P, -1)
    P = tf.add(P, 1)
    N = tf.subtract(tf.exp(N), 1)
    result = tf.add(P, N)
    return result

if __name__ == '__main__':
    a = [[1.0,-1.0,1.0],
         [-1.0,1.0,-1.0]]
    with tf.Session() as sess:
        result = sess.run(Miao(a))
        print(result)
