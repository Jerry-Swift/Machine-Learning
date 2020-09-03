#coding=utf-8
'''
搭建神经网络自动进行鸢尾花分类

'''
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets
'''
载入鸢尾花数据集的输入特征和类别标签
'''
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

'''
生成随机数种子，打乱鸢尾花数据顺序
'''
np.random.seed(10)
np.random.shuffle(x_data)
np.random.seed(10)
np.random.shuffle(y_data)
tf.random.set_seed(10)

'''
截取前120行数据作为训练接
后30行数据作为预测集
'''
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

'''
转换训练集和测试集数据类型为float32
否则在计算 y = tf.matmul(x_train, w1) + b1 的矩阵乘法中会报错
MatMul需要以float形式计算
'''
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

'''
对训练集和测试集进行分批次处理
'''
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

'''
取参数初值并设置为可训练
'''
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1))

lr = 0.1    #learn rate
train_loss_results = []
test_acc = []
epoch = 500     #迭代轮数
loss_all = 0

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):    #将以0~3的批次循环，每一批次包含32组数据（第3批次为24组）
        # print('step|x_train|y_train: {}|{}|{}'.format(step, x_train, y_train))
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1     #计算类别
            y = tf.nn.softmax(y)    #使y符合概率分布
            # print("y is : ", y)
            y_ = tf.one_hot(y_train, depth=3)   #将训练集中的实际值转换为独热编码
            # print('y_: ', y_)
            loss = tf.reduce_mean(tf.square(y_ - y))    #矩阵相减后求每个元素的平方值，再求矩阵所有元素的均方差，得到当前批次的损失值
            print('loss: ', loss)
            loss_all += loss.numpy()        #将一轮迭代下的四次批次计算出的损失值求均值，得到整轮迭代的损失值 numpy()使tensor对象转换为数值
            # print("loss.numpy: ", loss.numpy())
        grads = tape.gradient(loss, [w1, b1])   #计算损失函数中w1和b1的梯度
        # print('grads : ', grads)

        '''
        参数更新
        '''
        w1.assign_sub((lr * grads[0]))
        b1.assign_sub((lr * grads[1]))

    train_loss_results.append(loss_all / 4)     #一轮迭代的整体损失值
    loss_all = 0    #当前损失值归零，用于下一轮损失值计算


    '''
    预测部分
    '''
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) +b1
        y = tf.nn.softmax(y)
        # print('softmax(y) = ', y)
        pred = tf.argmax(y, axis=1)     #返回int64类型的每一行最大值对应的索引，相当于给出了所属类别，
        print('pred: ', pred)
        pred = tf.cast(pred, dtype=y_test.dtype)    #转换为int32类型
        print('pred.cast: ', pred)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)   #将预测的所有元素与测试集内元素一一对比，将true转换为int32类型（1）
        correct = tf.reduce_sum(correct)    #计算预测对的个数
        total_correct += int(correct)
        total_number += x_test.shape[0]     #样本行数30
        print('total_number1: ', total_number)
    acc = total_correct / total_number      #准确率 = 正确预测次数 / 样本数
    test_acc.append(acc)
    print('-------------------------')

'''
作图
'''
plt.title('Acc curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='$Accuracy$')
plt.legend()
plt.show()

plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label='$Loss$')
plt.legend()
plt.show()



