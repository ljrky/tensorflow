import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from libs.tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

np.random.seed(1)


"""
如何用TensorFlow框架写模型呢？

1，声明需要TensorFlow计算的变量(包括常量和变量，这些统称为Tensors)
2，定义需要进行的操作(计算)的函数
3，定义如何初始化变量的函数 (1,2,3就组成了一个计算图)
4，创建一个任务(称为Session)
5，使用任务一步步执行上面定义的函数
"""
# 声明需要TensorFlow计算的变量
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

# 定义需要进行的操作(计算)的函数
loss = tf.Variable((y - y_hat)**2, name='loss')

# 定义如何初始化变量的函数
init = tf.global_variables_initializer()

# 创建一个任务
with tf.Session() as session:
    # 使用任务一步步执行上面定义的函数
    session.run(init)
    # 使用任务一步步执行上面定义的函数
    print(session.run(loss))
    session.close()

# 定义变量
a = tf.constant(2)
b = tf.constant(10)
# 定义计算函数
c = tf.multiply(a,b)
print(c)
# 创建一个任务
session = tf.Session()
# 使用任务一步步执行上面定义的函数
print(session.run(c))

# 如何在任务执行的时候传递参数给任务
x = tf.placeholder(tf.int64, name="x")
print(session.run(2 * x, feed_dict={x : 3}))
session.close()
