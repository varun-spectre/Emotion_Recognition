#!/usr/bin/env python
# coding: utf-8


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
from typing import Tuple

import numpy as np
import scipy.io.wavfile as wav
from speechpy.feature import mfcc

mean_signal_length = 32000  # Empirically calculated for the given data set
import sys
from typing import Tuple
import glob
import numpy
from sklearn.metrics import accuracy_score, confusion_matrix
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

from sklearn.model_selection import train_test_split
from keras.utils import np_utils
_DATA_PATH = './SER/dataset1/'
_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")


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



def extract_data(flatten):
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS,flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), len(_CLASS_LABELS)

def get_feature_vector(file_path, flatten):
    return get_feature_vector_from_mfcc(file_path, flatten, mfcc_len=39)


import numpy as np
import tensorflow as tf
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion
import os



def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot



def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    #tf.cast(inputs, tf.float64)
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]],dtype=tf.float64))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]],dtype=tf.float64))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]],dtype=tf.float64), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]],dtype=tf.float64), trainable=False)

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



def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1,dtype=tf.float64))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1,dtype=tf.float64))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1,dtype=tf.float64))

    # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
    #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
    #v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    v = tf.sigmoid(tf.tensordot(inputs, W_omega, axes=1) + b_omega)
    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
    alphas = tf.nn.softmax(vu)              # (B,T) shape also

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
    


epsilon = 1e-3
def leaky_relu(x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')



def my_model(features, labels, mode):
    #inputs
    # Input Layer
    inputs = tf.reshape(features["x"], [-1, 198, 39, 1])
    #tf.cast(inputs, tf.float32)
    num_classes=4
    is_training=True
    L1=128
    L2=256
    cell_units=128
    num_linear=768
    p=9
    time_step=99
    F1=64
    dropout_keep_prob=1
    layer1_filter = tf.get_variable('layer1_filter', shape=[5, 3, 1, L1], dtype=tf.float64, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer1_bias = tf.get_variable('layer1_bias', shape=[L1], dtype=tf.float64,
                                  initializer=tf.constant_initializer(0.1))
    layer1_stride = [1, 1, 1, 1]
    layer2_filter = tf.get_variable('layer2_filter', shape=[5, 3, L1, L2], dtype=tf.float64, 
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    layer2_bias = tf.get_variable('layer2_bias', shape=[L2], dtype=tf.float64,
                                  initializer=tf.constant_initializer(0.1))
    layer2_stride = [1, 1, 1, 1]
    
    linear1_weight = tf.get_variable('linear1_weight', shape=[p*L2,num_linear], dtype=tf.float64,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    linear1_bias = tf.get_variable('linear1_bias', shape=[num_linear], dtype=tf.float64,
                                  initializer=tf.constant_initializer(0.1))
 
    fully1_weight = tf.get_variable('fully1_weight', shape=[2*cell_units,F1], dtype=tf.float64,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully1_bias = tf.get_variable('fully1_bias', shape=[F1], dtype=tf.float64,
                                  initializer=tf.constant_initializer(0.1))
    fully2_weight = tf.get_variable('fully2_weight', shape=[F1,num_classes], dtype=tf.float64,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
    fully2_bias = tf.get_variable('fully2_bias', shape=[num_classes], dtype=tf.float64,
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
    print(linear1.get_shape())
    linear1 = tf.reshape(linear1, [-1, time_step, num_linear])
    print(linear1.get_shape())
    gru_fw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    gru_bw_cell1 = tf.contrib.rnn.BasicLSTMCell(cell_units, forget_bias=1.0)
    outputs1, output_states1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell1,cell_bw=gru_bw_cell1,inputs= linear1,dtype=tf.float64,time_major=False,scope='LSTM1')
    gru, alphas = attention(outputs1, 1, return_alphas=True)
    fully1 = tf.matmul(gru,fully1_weight) + fully1_bias
    fully1 = leaky_relu(fully1, 0.01)
    fully1 = tf.nn.dropout(fully1, dropout_keep_prob)
    Ylogits = tf.matmul(fully1, fully2_weight) + fully2_bias 
    
    #compute predictions
    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=Ylogits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(Ylogits, name="softmax_tensor")
      }
    
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=Ylogits)
    accuracy =  tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    # Configure the Training Op (for TRAIN mode)
    logging_hook = tf.train.LoggingTensorHook({"loss" : loss, "accuracy" : accuracy[1]}, every_n_iter=10)
    
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
      }
    
    starter_learning_rate = 0.001
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(starter_learning_rate, tf.train.get_global_step(), 2000, 0.96, staircase=True)
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,training_hooks = [logging_hook])
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    
   
    



def train(run_config):
    x_train, x_test, y_train, y_test, num_labels = extract_data(flatten=False)
    x_train = np.reshape(x_train,(-1,198,39,1))
    x_test = np.reshape(x_test,(-1,198,39,1)) 
    y_train = np.reshape(y_train,(-1,1))
    y_train = np.asarray(y_train)
    y_test = np.reshape(y_test,(-1,1))
    y_test = np.asarray(y_test)
    #shuffling data
    p = np.random.permutation(len(x_train))
    x_train = x_train[p]
    y_train = y_train[p]
#     y_train = dense_to_one_hot(y_train,FLAGS.num_classes)
#     y_test = dense_to_one_hot(y_test,FLAGS.num_classes)
    tf.logging.set_verbosity(tf.logging.INFO)
    train_steps = 10000
    eval_steps = 7 
    emotion_classifier = tf.estimator.Estimator(model_fn=my_model,model_dir="./summaries/",config = run_config)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},y=y_train,batch_size=100,num_epochs=None,shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_test},y=y_test,shuffle=False)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=train_steps)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=eval_steps, throttle_secs=120)
    return tf.estimator.train_and_evaluate(emotion_classifier, train_spec, eval_spec)



# In[15]:


def main():
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.estimator.RunConfig(session_config=sess_config,model_dir="./summaries/", keep_checkpoint_max=200,
                                    log_step_count_steps=10, save_checkpoints_steps =200)
    train(config)




main()




