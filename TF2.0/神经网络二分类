#coding=utf-8

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv('.\class2\class2\dot.csv')
x_data = np.array(df[['x1', 'x2']])
y_data = np.array(df['y_c'])

x_train = np.vstack(x_data).reshape(-1, 2)  #reshape(-1,2) 行数未知，列数设置为2
print('x_data.vstack: ', np.vstack(x_data))
print('x_train.reshape: ', x_train)
y_train = np.vstack(y_data).reshape(-1, 1)
print('y_data.vstack: ', np.vstack(y_data))
print('y_train.reshape:', y_train)

Y_c = [['red' if y else 'blue'] for y in y_train]
print(Y_c)

x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=[11]))

w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=[1]))

lr = 0.01
epoch = 500

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            h1 = tf.nn.relu(h1)
            y = tf.matmul(h1, w2) + b2

            loss = tf.reduce_mean(tf.square(y_train - y))

        variables = [w1 ,b1 ,w2, b2]
        grads = tape.gradient(loss , variables)

        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])


    if epoch % 20 == 0:
        print('epoch: ', epoch, 'loss:', float(loss) )

print('---------------------')

xx, yy = np.mgrid[-3:3:.1, -3:3:.1]     #生成[-3,3]之间0.01步长的数字
print('xx: {},yy: {}'.format(xx.shape, yy.shape))
grid = np.c_[xx.ravel(), yy.ravel()]    #ravel()将数组展开为一维形式
grid = tf.cast(grid, tf.float32)

probs = []

for x_test in grid:
    h1 = tf.matmul([x_test], w1) + b1
    h1 = tf.nn.relu(h1)
    y = tf.matmul(h1, w2) + b2
    probs.append(y)

x1 = x_data[:, 0]
x2 = x_data[:, 1]

probs = np.array(probs).reshape(xx.shape)
plt.scatter(x1, x2, color=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()

