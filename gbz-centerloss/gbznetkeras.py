from keras.layers import Conv2D,AveragePooling2D,MaxPooling2D
from keras.layers import  Dense, BatchNormalization, Dropout, Activation,regularizers
from keras.layers.merge import concatenate
import keras as k
import tensorflow as tf

class my_alex(object):
    def __init__(self, x, keep_prob, num_class):
        self.x = x
        self.keep_prob = keep_prob
        self.num_class = num_class
        self.init = k.initializers.glorot_normal()
        self.regular = regularizers.l1_l2(l1=0.1, l2=0.5)
    def predict(self):
        net1 = Conv2D(128, (5, 5), padding='same', strides=[2, 2],activation='relu',
                      kernel_initializer=self.init)(self.x)
        net1 = MaxPooling2D((3, 3), strides=(2, 2))(net1)
        
        net2 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],activation='relu',
                      kernel_initializer=self.init)(net1)
        net2 = MaxPooling2D((3, 3), strides=(2, 2))(net2)
       
        net3 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],activation='relu',
                      kernel_initializer=self.init)(net2)
        net3 = MaxPooling2D((3, 3), strides=(2, 2))(net3)
        net4 = Conv2D(256, (3, 3), padding='same', strides=[1, 1],activation='relu',
                      kernel_initializer=self.init)(net3)

        x = concatenate([net3, net4], axis=3)
        
        net7 = tf.reshape(x, [-1, 7*2*512])
        # net8 = Dense(4096, activation='sigmoid',activity_regularizer=regularizers.l2(0.01))(net7)
        net8 = Dense(1024, activation='sigmoid')(net7)
        net9 = tf.nn.dropout(net8, self.keep_prob)
        # net9 = Dense(4096, activation='sigmoid',activity_regularizer=regularizers.l1(0.01))(net9)
        # net9 = Dense(512, activation='sigmoid')(net9)
        net10 = Dense(self.num_class)(net9)
        return net10,net8
