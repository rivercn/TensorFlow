import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
from resnet_data_read import *

def weight_init(shape):
    init = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

def Miao_activate_function(feature):
    '''
    :param feature:
    :return:

    '''
    N = feature*(-1)
    N= tf.nn.relu(N)
    N = -1*tf.exp(N)+1
    P = feature
    P= tf.nn.relu(P)
    P = tf.exp(P)-1
    result = tf.add(P,N)
    return result

def identity_block(input,
                   kernel_size,
                   in_filter_nums,
                   out_filter_nums,
                   stage,
                   block,
                   training=True):
    '''
    :param input: 输入的特征
    :param kernel_size: 卷积核的尺寸
    :param in_filter_nums:输入的卷积核的数目
    :param out_filter_nums: 输出的卷积核的数目
    :param stage:阶段
    :param block:块
    :param training:是否训练 true or false
                     由于去掉了BN层，因此此层暂时不用
    :return:特征值

    *注：
    1、没有使用BN层
    2、使用的激活函数不是Relu
    '''
    block_name = 'Resnet'+str(stage)+block
    f1,f2,f3 = out_filter_nums
    with tf.variable_scope(block_name):
        shorcut = input
        '''
        f1,f2,f3主要的作用就是因为一个残差块有三层，每层的卷积核数目不一样
        '''
        f1,f2,f3 = out_filter_nums

        #first
        conv1 = tf.layers.conv2d(input,f1,kernel_size=[1,1],strides=[1,1],padding='SAME')
        #conv1 = tf.nn.relu(conv1)
        conv1 = Miao_activate_function(conv1)

        #second
        conv2 = tf.layers.conv2d(conv1,f2,kernel_size=[kernel_size,kernel_size],
                                 strides=[1,1],padding='SAME')
        conv2 = Miao_activate_function(conv2)

        #third
        conv3 = tf.layers.conv2d(conv2, f3, kernel_size=[1, 1],
                                 strides=[1, 1], padding='VALID')

        #final
        add = tf.add(conv3,shorcut)
        add = Miao_activate_function(add)
    return  add

def convolutional_block(input,
                        kernel_size,
                        in_filter_nums,
                        out_filter_nums,
                        stage,
                        block,
                        stride = 2,
                        training=True):
    '''
    :param input: 输入的特征
    :param kernel_size: 卷积核的尺寸
    :param in_filter_nums:输入的卷积核的数目
    :param out_filter_nums: 输出的卷积核的数目
    :param stage:阶段
    :param block:块
    :param stride:步长
    :param training:是否训练 true or false
                     由于去掉了BN层，因此此层暂时不用
    :return:特征值

    *注：
    1、没有使用BN层
    2、使用的激活函数不是Relu
    '''

    block_name = "Resnet"+str(stage)+block
    with tf.variable_scope(block_name):
        shortcut = input
        f1,f2,f3 = out_filter_nums

        #first
        conv1 = tf.layers.conv2d(input,f1,kernel_size=[1,1],
                                 strides=[stride,stride],padding='VALID')
        conv1 = Miao_activate_function(conv1)

        #second
        conv2 = tf.layers.conv2d(conv1,f2,kernel_size=[kernel_size,kernel_size],
                                 strides=[1,1],padding="SAME")
        conv2 = Miao_activate_function(conv2)

        #third
        conv3 = tf.layers.conv2d(conv2,f3,kernel_size=[1,1],
                                 strides=[1,1],padding="VALID")
        conv3 = Miao_activate_function(conv3)

        #shortPath
        shortcut = tf.layers.conv2d(shortcut,f3,kernel_size = [1,1],
                                    strides=[stride,stride],padding="VALID")

        #final
        add = tf.add(shortcut,conv3)
        add = Miao_activate_function(add)

    return add

def deepnn(input):
    x = tf.pad(input,tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]),"constant")
    with tf.variable_scope('reference'):
        #因为没有BN层，所以下面这一条的注释掉，
        #如果加上了BN层，则需要区分是否在训练还是在测试
        #training =  tf.placeholder(tf.bool,name='training')

        #stage 1

        x = tf.layers.conv2d(input,64,kernel_size=[3,3],strides=[2,2],padding='VALID')
        x = Miao_activate_function(x)
        x = tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')


        '''
        convolutional_block中
        shortcut要进行降维
        
        identity_block中
        shortcut不需要进行降维操作
        '''
        #stage 2
        x = convolutional_block(x,3,64,[64,64,256],2,'a',stride=1)
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c')

        # stage 3
        x = convolutional_block(x, 3, 256, [128, 128, 512], 3, 'a')
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'b')
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'c')
        x = identity_block(x, 3, 512, [128, 128, 512], 3, 'd')

        # stage 4
        x = convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'e')
        x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f')

        # stage 5
        x = convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a')
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b')
        x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c')

        x = tf.nn.avg_pool(x,[1,2,2,1],strides=[1,1,1,1],padding='VALID')

        flatten = tf.layers.flatten(x)
        x = tf.layers.dense(flatten,units=50)
        x= Miao_activate_function(x)
        #Droput 按比例失活神经元，防止过拟合

        with tf.name_scope('Droupout'):
            keep_prob = tf.placeholder(tf.float32)
            x = tf.nn.dropout(x,keep_prob)

        logits = tf.layers.dense(x,units=6,activation=tf.nn.softmax)

    return logits,keep_prob

def weight_varible(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def train(X,Y):
    features = tf.placeholder(tf.float32,[None,64,64,3])
    labels = tf.placeholder(tf.int64,[None,6])

    logits,keep_prob = deepnn(features)

    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels,logits= logits)
    cross_entropy = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    mini_batches =random_mini_batches(X,Y,mini_batch_size=32)

    saver = tf.train.Saver()
    with tf.Session() as  sess:
        sess.run(tf.global_variables_initializer())
        for i in  range(1000):
            x,y = mini_batches[np.random.randint(0,len(mini_batches))]
            train_op.run(feed_dict={
                features:x,labels:y,keep_prob:0.5,
            })

            if i%20==0:
                loss = sess.run(cross_entropy,feed_dict={
                    features:x,labels:y,keep_prob:0.5
                })
                print("step %d loss %d"%(i,loss))
        saver.save(sess,".\\model_saving\\1")




if __name__ == '__main__':
    '''
    a = tf.constant([[1, 1, 1], [-1, -1, -1]], dtype=tf.float32)
    b = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    with tf.Session() as sess:
        result = sess.run(tf.nn.relu(a))
        print(result)
    data = h5py.File('.\\h5\\train_signs.h5','r')
    print(data)
    data_dir = ".\\h5"
    data = load_dataset(data_dir)
    X,Y,_,_ = process_orig_datasets(data)
    print(X)
    print(Y)
    train(X,Y)
    '''
    a = tf.constant([[1, 1, 1], [-1, -1, -1]], dtype=tf.float32)
    b = tf.constant([[1, 1, 1], [2, 2, 2]], dtype=tf.float32)
    with tf.Session() as sess:
        result = sess.run(tf.nn.relu(a))
        Miao = sess.run(Miao_activate_function(a) )
        print(result)
        print(Miao)

