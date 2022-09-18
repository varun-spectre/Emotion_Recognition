#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


import sys
import keras
import numpy as np
from keras import Sequential
from keras.layers import  Dense, Dropout, Conv2D, Flatten,BatchNormalization, Activation, MaxPooling2D
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import merge


# In[7]:


from sklearn.model_selection import train_test_split
from keras.utils import np_utils
_DATA_PATH = './SER/dataset1/'
_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")


# In[8]:


def extract_data(flatten):
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS,flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(data,labels,test_size=0.2,random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test), len(_CLASS_LABELS)

def get_feature_vector(file_path, flatten):
    return get_feature_vector_from_mfcc(file_path, flatten, mfcc_len=39)


# In[9]:


def predict_one(sample):
    return np.argmax(model.predict(np.array([sample])))

def predict(samples):
    results = []
    for _, sample in enumerate(samples):
        results.append(predict_one(sample))
    return tuple(results)


# In[11]:


class_weight = {0: 1.,
                1: 1.,
                2: 2.,
                3: 1.}


# In[10]:


model  = Sequential()
model.add(Bidirectional(LSTM(128,input_shape=(198, 39))))

# model = Sequential()
#model.add(keras.layer.merge([model2, model1], mode='sum'))
# added = keras.layers.merge.Add()([model2,model1])
# out = keras.layers.Dense(32)(added)
# model = keras.models.Model(inputs=[input1, input2], outputs=out)
#added = keras.layers.Add()([x1, x2])

model.add(Dropout(0.5))
#model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
n_epochs = 50
best_acc = 0
save_path1 =  './models/BilSTMEpoch_model.h5'
x_train, x_test, y_train, y_test, num_labels = extract_data(flatten=False)
y_train = np_utils.to_categorical(y_train)
y_test_train = np_utils.to_categorical(y_test)
x_val = x_test
y_val = y_test_train
#shuffle data
p = np.random.permutation(len(x_train))
x_train = x_train[p]
y_train = y_train[p]
model.fit(x_train, y_train,validation_data=(x_val,y_val), batch_size=32,epochs=50,class_weight= class_weight)
#print("Epoch Number:{}".format(i))
#     loss, acc = model.evaluate(x_val, y_val)
#     if acc > best_acc:
#         best_acc = acc
#         model.save_weights(save_path1)
        
#print(type(x_test))        
#model.evaluate(x_test, y_test)
predictions = predict(x_test)
print(y_test)
print(predictions)
print('Accuracy:%.3f\n' % accuracy_score(y_pred=predictions,
                                         y_true=y_test))
print('Confusion matrix:', confusion_matrix(y_pred=predictions,
                                            y_true=y_test))
print(model.summary(), file=sys.stderr)
save_path =  'BilSTM1_model.h5'
model.save_weights(save_path)


# In[ ]:




