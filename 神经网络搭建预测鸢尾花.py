import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

'''
载入鸢尾花数据特征和标签
'''
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

'''
打乱数据顺序
'''
np.random.seed(10)
np.random.shuffle(x_data)
np.random.seed(10)
np.random.shuffle(y_data)
tf.random.set_seed(10)

'''
前120个数据对用作训练集
后30个用作测试集
'''
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

'''
将训练集转换为float32类型，否则矩阵相乘时会报错
'''
x_train = tf.cast(x_train, tf.float32)
y_train = tf.cast(y_train, tf.float32)

'''
对训练集和测试集进行分批
'''
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

'''
具有4个特征，因此搭建4个输入节点
结果分为3类，搭建3个神经元
'''
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1))

lr = 0.1    #学习速率
train_loss_results = []     #记录每一轮的loss值
test_acc = []       #记录每一轮的acc值
epoch = 500
loss_all = 0

'''
训练神经网络
'''
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1     #神经网络乘加运算
            y = tf.nn.softmax(y)    #使输出值符合概率分布
            y_ = tf.one_hot(y_train, depth=3)   #将标签转换为独热编码，方便计算loss和acc
            loss = tf.reduce_mean(tf.square(y_ - y))    #损失函数计算
            loss_all += loss.numpy()    #将每一步的loss值累加并计算均值，计算更准确的loss值

        grades = tape.gradient(loss, [w1, b1])  #计算loss对w1和b1的梯度

        '''
        梯度更新
        '''
        w1.assign_sub(lr * grades[0])
        b1.assign_sub(lr * grades[1])

    print("Epoch {}, loss: {}").format(epoch, loss_all/4)
    train_loss_results.append(loss_all / 4)
    loss_all = 0

    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]

    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc: ", acc)
    print("------------------------")

plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()




