
from my_newDataGenerator import ImageDataGenerator
import tensorflow as tf
import os
# from gbznetkeras  import my_alex
# from footnet_v5  import my_alex
from nettest  import my_alex
import numpy as np
import matplotlib.pyplot as plt
from center  import center
#训练轮数
epoch = 3
#每个batch的大小
batch_size = 1
#学习率
learning_rate = 0.01
#分类个数
num_class = 1780
center_loss_weight=0.03
alpha=0.6
#路径设置
train_file = 'F:\pywork\database\\footweight\\recognition\dongbo\\temp\V1.4.0.3\\NO3\\train.txt'
val_file = 'F:\pywork\database\\footweight\\recognition\dongbo\\temp\V1.4.0.3\\NO3\\testall.txt'
checkpoint_path ='F:\pywork\database\\footweight\\recognition\out'

filewriter_path = os.path.join(checkpoint_path, 'writer')
#训练好的模型
filewriterpath = 'F:\pywork\database\\footweight\\recognition\out\\my_net_epoch984.ckpt'
#初始化占位，x：输入图像，y：输入图像标签
x = tf.placeholder(tf.float32, [batch_size, 128, 59, 3])
y = tf.placeholder(tf.float32, [batch_size, num_class])
labels = tf.placeholder(tf.int64, [batch_size])
keep_prob = tf.placeholder(tf.float32,shape=None)
test = tf.placeholder(tf.bool,shape=None)
#从my_net类中导入定义好的网络结构
model = my_alex(x, keep_prob,num_class)
#返回各卷积层和最后的分类结果
y_out,net6= model.predict()

center_loss, centers_batch,centers=center.get_center_loss(net6, labels, alpha, num_class)
#计算loss函数
with tf.name_scope('loss'):
#先计算softmax(y_out)再计算交叉熵，然后按batch求平均
  softmaxloss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=y_out,labels=y))
  loss = tf.add(softmaxloss, center_loss * center_loss_weight)
#对loss函数值汇总并记录到日志中（存放在protobuf中），供tensorboard读取
tf.summary.scalar('loss', loss)

#构造一个使用Adadelta算法的优化器
#opt1 = tf.train.AdadeltaOptimizer(learning_rate, rho=0.9, epsilon=1e-6)
opt1=tf.train.GradientDescentOptimizer(learning_rate)
#最小化loss函数
opt = opt1.minimize(loss)

# 计算准确度得到一组布尔值
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_out, axis=1))
with tf.name_scope('accuracy'):
#将布尔值转化为浮点数求均值
 accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accauracy',accuracy)

#数据集准备
train_generator = ImageDataGenerator(train_file, horizontal_flip = False, shuffle = True)
val_generator = ImageDataGenerator(val_file, shuffle = True)


#需要训练的变量列表
varlist = tf.trainable_variables()
#直接记录变量var的直方图，输出带直方图的汇总的protobuf
for var in varlist:
    tf.summary.histogram(var.name, var)
#计算varlist相关的梯度
gradients = opt1.compute_gradients(loss, var_list=varlist)
for gradient, var in gradients:
  tf.summary.histogram(var.name + '/gradient', gradient)


#将上面所有类型的汇总再进行一次合并
merged_summary = tf.summary.merge_all()
#将汇总的protobuf写入到event文件中
writer = tf.summary.FileWriter(filewriter_path)
#保存模型
saver = tf.train.Saver()
acc_max = 0
#创建会话
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,filewriterpath)
    #计算训练和测试的batch_size的个数
    num_step_train = np.floor(train_generator.data_size / batch_size).astype(np.int16)
    num_step_test= np.floor(val_generator.data_size / batch_size).astype(np.int16)
    print('num_step_train', num_step_train,'num_step_test',num_step_test)
    #添加一个 Graph到事件文件中 to the event file
    writer.add_graph(sess.graph)
    #定义向量
    Cost = []
    test_acc0 = np.zeros(int(np.ceil(epoch)))
    test_loss0 = np.zeros(int(np.ceil(epoch)))
    for i in range(epoch):

        #开始测试
        step_test = 0
        print('testing in Test')
        test_acc = 0
        test_loss=0
        while step_test < num_step_test:
            batch_x, batch_y,batch_labels = val_generator.next_batch(batch_size)
            #计算每个batch的正确率
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y,labels:batch_labels,keep_prob:1.0})
            tlos = sess.run(loss, feed_dict={x: batch_x, y: batch_y,labels:batch_labels, keep_prob: 1.0})
            test_loss = test_loss + tlos
            #print('num_step_test_accuracy=', acc)
            test_acc = test_acc+acc
            step_test = step_test+1
        #输出所有batch的正确率的均值
        print('test_accuracy=', test_acc/step_test)
        print('test_loss=', test_loss / step_test)
        val_generator.reset_pointer()
        train_generator.reset_pointer()
        #保存每轮训练的平均正确率
        test_acc0[i] = test_acc/step_test
        test_loss0[i] = test_loss / step_test
        #保存正确率最大的模型

    print(test_acc0)
    A = np.sum(test_acc0)
    #所有测试正确率均值
    C = A / epoch
    #所有测试正确率最大值
    D = np.max(test_acc0)
    print('ave_acc', C, 'max_acc', D)
    Cost[0]=2
# 代价函数曲线
    f1 = plt.figure(1)
    plt.plot(Cost)
    plt.grid()
    plt.show()
    f2 = plt.figure(2)
    plt.plot(test_loss0)
    plt.grid()
    plt.show()
    # 准确率曲线
    f3 = plt.figure(3)
    plt.plot(test_acc0)
    plt.grid()
    plt.show()




