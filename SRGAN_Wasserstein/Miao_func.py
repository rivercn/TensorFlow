import tensorflow as tf
import numpy as  np
from tensorflow.python.framework import ops





def Miao(feature):
    P = (feature+np.abs(feature))/2
    N = np.abs(feature -P)
    P = np.exp(P)-1
    N = np.exp(N)*-1+1
    result = P+N
    return result

def Miao_grad(x):
    return np.exp(np.abs(x))

Miao_np = np.vectorize(Miao)
Miao_grad_np = np.vectorize(Miao_grad)

Miao_np_32 = lambda x: Miao_np(x).astype(np.float32)
Miao_grad_np_32 = lambda x: Miao_grad_np(x).astype(np.float32)

def Miao_grad_tf(x, name=None):
    #with tf.name_scope(name, "Miao_grad_tf", [x]) as name:
        y = tf.py_func(Miao_grad_np_32, [x], [tf.float32], name=name, stateful=False)
        return y[0]

def my_py_func(func, inp, Tout, stateful=False, name=None, my_grad_func=None):
    # need to generate a unique name to avoid duplicates:
    random_name = "PyFuncGrad" + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(random_name)(my_grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": random_name, "PyFuncStateless": random_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def _Miao_grad(op,pred_grad):
    x = op.inputs[0]
    cur_grad = Miao_grad(x)
    next_grad = pred_grad * cur_grad
    return next_grad

def Miao_tf(x,name = None):
   # with tf.name_scope(name,"Miao_tf",[x]) as name:
        y = my_py_func(Miao_np_32,
                       [x],
                       [tf.float32],
                       stateful=False,
                       name=name,
                       my_grad_func=_Miao_grad
                       )
        return y[0]

if __name__ == '__main__':
    a = [[1,-1,1],
         [-1,1,-1]]
    print(Miao(a))
