#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
import os
import sys
from typing import Tuple


import matplotlib.pyplot as plt

import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc

mean_signal_length = 32000  # Empirically calculated for the given data set
import sys
from typing import Tuple
import glob
import numpy
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
#from acrnn1 import acrnn
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
_DATA_PATH = './dataset1/'
_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")

def classLabels(s):
    if(s=='neu'):
        return 0
    elif(s=='ang'):
        return 1
    elif(s=='hap'):
        return 2
    elif(s=='sad'):
        return 3
    else:
        return -1
rootdir = '/home/varun/Downloads/speech-emotion-recognition-master/IEMOCAP_full_release'    



"""

Datasets used- Ravdess + Savee + EmoDB(Berlin dataset) + IEMOCAP
sample rate- 16000
channels- mono

"""

def get_feature_vector_from_mfcc(file_path: str, flatten: bool,
                                 mfcc_len: int = 39) -> np.ndarray:
    fs, signal = wav.read(file_path)
    s_len = len(signal)
    # pad the signals to have same size if lesser than required
    # else slice them
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                        'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    if flatten:
        # Flatten the data
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients


# In[3]:


def get_data(data_path: str, flatten: bool = True, mfcc_len: int = 39,
             class_labels: Tuple = ("Neutral", "Angry", "Happy", "Sad")) -> \
        Tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []
    names = []
    #Procces Ravdess + Savee + EmoDB
    cur_dir = os.getcwd()
    sys.stderr.write('curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("started reading folder %s\n" % directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            filepath = os.getcwd() + '/' + filename
            feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
                                                          mfcc_len=mfcc_len,
                                                          flatten=flatten)
#             print(feature_vector.shape)
            data.append(feature_vector)
            labels.append(i)
            names.append(filename)
        sys.stderr.write("ended reading folder %s\n" % directory)
        os.chdir('..')
    os.chdir(cur_dir)
    
    #Process IEMOCAP
    for speaker in os.listdir(rootdir):
        if(speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir,speaker,'sentences/wav')
            emoevl = os.path.join(rootdir,speaker,'dialog/EmoEvaluation')
            for sess in os.listdir(sub_dir):
                if(sess[7] == 'i'):
                    emotdir = emoevl+'/'+sess+'.txt'
                    #emotfile = open(emotdir)
                    emot_map = {}
                    with open(emotdir,'r') as emot_to_read:
                        while True:
                            line = emot_to_read.readline()
                            if not line:
                                break
                            if(line[0] == '['):
                                t = line.split()
                                emot_map[t[3]] = t[4]
                                
        
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        #wavname = filename[-23:-4]
                        wavname = filename.split("/")[-1][:-4]
                        emotion = emot_map[wavname]
                        if(emotion in ['hap','ang','neu','sad']):
                            filepath = filename
                            feature_vector = get_feature_vector_from_mfcc(file_path=filepath,
                                                          mfcc_len=mfcc_len,
                                                          flatten=flatten)
                            data.append(feature_vector)
                            labels.append(classLabels(emotion))
                            names.append(filename)    
    
    
    return np.array(data), np.array(labels)






# In[6]:


def extract_data(flatten):
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS,flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), len(_CLASS_LABELS)

def get_feature_vector(file_path, flatten):
    return get_feature_vector_from_mfcc(file_path, flatten, mfcc_len=39)





# In[10]:




# In[11]:


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



# In[18]:


tf.app.flags.DEFINE_integer('num_epoch', 1000, 'The number of epoches for training.')
tf.app.flags.DEFINE_integer('num_classes', 4, 'The number of emotion classes.')
tf.app.flags.DEFINE_integer('batch_size', 59, 'The number of samples in each batch.')
tf.app.flags.DEFINE_boolean('is_adam', True,'whether to use adam optimizer.')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'learning rate of Adam optimizer')
tf.app.flags.DEFINE_float   ('dropout_keep_prob',     1,        'the prob of every unit keep in dropout layer')
tf.app.flags.DEFINE_string  ('model_name', 'model4.ckpt', 'model name')
tf.app.flags.DEFINE_string('f', '', 'kernel')

FLAGS = tf.app.flags.FLAGS


# In[19]:


def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)



# In[21]:


epsilon = 1e-3

def leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def acrnn(inputs, num_classes=4,is_training=True,L1=128,L2=256,cell_units=128,
    num_linear=768,p=9,time_step=99,F1=64,dropout_keep_prob=1):
    layer1_filter = tf.get_variable('layer1_filter', shape=[5, 3, 1, L1], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer1_bias = tf.get_variable('layer1_bias', shape=[L1], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer1_stride = [1, 1, 1, 1]
    layer2_filter = tf.get_variable('layer2_filter', shape=[5, 3, L1, L2], dtype=tf.float32, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer2_bias = tf.get_variable('layer2_bias', shape=[L2], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    layer2_stride = [1, 1, 1, 1]
    
    linear1_weight = tf.get_variable('linear1_weight', shape=[p*L2,num_linear], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    linear1_bias = tf.get_variable('linear1_bias', shape=[num_linear], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
 
    fully1_weight = tf.get_variable('fully1_weight', shape=[2*cell_units,F1], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully1_bias = tf.get_variable('fully1_bias', shape=[F1], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    fully2_weight = tf.get_variable('fully2_weight', shape=[F1,num_classes], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully2_bias = tf.get_variable('fully2_bias', shape=[num_classes], dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
    
    layer1 = tf.nn.conv2d(inputs, layer1_filter, layer1_stride, padding='SAME')
    layer1 = tf.nn.bias_add(layer1,layer1_bias)
    layer1 = leaky_relu(layer1, 0.01)
    print(layer1.get_shape())
    layer1 = tf.nn.max_pool(layer1,ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID', name='max_pool')
    layer1 = tf.contrib.layers.dropout(layer1, keep_prob=dropout_keep_prob, is_training=is_training)
    print(layer1.get_shape())
    layer2 = tf.nn.conv2d(layer1, layer2_filter, layer2_stride, padding='SAME')
    layer2 = tf.nn.bias_add(layer2,layer2_bias)
    layer2 = leaky_relu(layer2, 0.01)
    layer2 = tf.contrib.layers.dropout(layer2, keep_prob=dropout_keep_prob, is_training=is_training)
    print(layer2.get_shape())
    layer2 = tf.reshape(layer2,[-1,time_step,L2*p])
    layer2 = tf.reshape(layer2, [-1,p*L2])
    linear1 = tf.matmul(layer2,linear1_weight) + linear1_bias
    linear1 = batch_norm_wrapper(linear1,is_training)
    linear1 = leaky_relu(linear1, 0.01)
    linear1 = tf.reshape(linear1, [-1, time_step, num_linear])
    gru_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    gru_bw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    outputs1, output_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell1,cell_bw=gru_bw_cell1,inputs= linear1,dtype=tf.float32,time_major=False,scope='LSTM1')
    outputs1 = tf.concat(outputs1, 2)
    fully1 = tf.matmul(outputs1,fully1_weight) + fully1_bias
    fully1 = leaky_relu(fully1, 0.01)
    fully1 = tf.nn.dropout(fully1, dropout_keep_prob)
    Ylogits = tf.matmul(fully1, fully2_weight) + fully2_bias
    #Ylogits = tf.nn.softmax(Ylogits)
    return Ylogits
#     print(Ylogits.get_shape())


# In[22]:


# with tf.Session() as sess:
#     inputs=tf.random_normal((885, 198, 39, 1))
#     out = acrnn(inputs)
#     init = tf.global_variables_initializer()
#     sess.run(init)
# #     print(sess.run(out.get_shape()))


# In[23]:





# In[24]:


def train():
    x_train, x_test, y_train, y_test, num_labels = extract_data(flatten=False)
    x_train = np.reshape(x_train,(-1,198,39,1))
    x_test = np.reshape(x_test,(-1,198,39,1)) 
    y_train = np.reshape(y_train,(-1,1))
    y_train = np.asarray(y_train)
    y_test = np.reshape(y_test,(-1,1))
    y_test = np.asarray(y_test)
    test_size = x_test.shape[0]
    dataset_size = x_train.shape[0]
    best_test_uw = 0
    y_train = dense_to_one_hot(y_train,FLAGS.num_classes)
    y_test = dense_to_one_hot(y_test,FLAGS.num_classes)
    X = tf.placeholder(tf.float32,shape=[None,198,39,1])
    Y = tf.placeholder(tf.int32,shape=[None,4])
    is_training = tf.placeholder(tf.bool)
    lr = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    Ylogits = acrnn(X, is_training=is_training, dropout_keep_prob=keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels =  Y, logits =  Ylogits)
    cost = tf.reduce_mean(cross_entropy)
    var_trainable_op = tf.trainable_variables
    
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)  
    correct_pred = tf.equal(tf.argmax(Ylogits, 1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver=tf.train.Saver(tf.global_variables())
    #code for test accuracy
#     Ylogits1 = acrnn(X, is_training=False, dropout_keep_prob=1)
#     correct_pred1 = tf.equal(tf.argmax(Ylogits1, 1), tf.argmax(Y,1))
#     accuracy1 = tf.reduce_mean(tf.cast(correct_pred1, tf.float32))
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        loss_list = []
        for i in range(11):
            start = (i * FLAGS.batch_size) % dataset_size
            end = min(start+FLAGS.batch_size, dataset_size)
            [_,tcost,tracc] = sess.run([train_op,cost,accuracy], feed_dict={X:x_train[start:end,:,:,:], Y:y_train[start:end,:],is_training:True, keep_prob:FLAGS.dropout_keep_prob, lr:FLAGS.learning_rate})
            if(i%5==0):
                print("*************")
                print("tcost")
                print(tcost)
                print("tracc")
                print(tracc)
                loss_list.append(tcost)
        myarray = np.asarray(loss_list)
        plt.plot(myarray)
        plt.show()
        acc = sess.run(accuracy,feed_dict={X:x_test[start:end,:,:,:], Y:y_test[start:end,:],is_training:False})
        print(acc)
        



if __name__=='__main__':
    train()


# In[ ]:




