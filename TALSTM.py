#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.layers import LSTM,RepeatVector,Dense,Activation,Add,Reshape,Input,Lambda,Multiply,Concatenate,Dot,Merge
from sklearn.model_selection import train_test_split
from keras.models import Model
import numpy as np
import pandas as pd
import os
import h5py
from code.layer_definition import My_Transpose,My_Dot
from code.trend_fuction import get_trend
####参数设置
T = 25  #时间序列窗口大小
n = 12 #输入特征个数
m =n_h = n_s = 20  #length of hidden state m
p = n_hde0 = n_sde0 = 30  #p
path = 'data/'
data_path = path+'000002.csv'
cache_path = os.path.join(path,'cache')
fname_model = 'model_T{}.h5'.format(T)
CACHEDATA = False
batch_size = 128
epochs = 100
test_split = 0.2
mothed='MACD'
########################数据预处理#######################
def get_data(data_path, T, mothed='MACD'):
    input_X = []
    input_Y = []
    label_Y = []
    trend_Y = []
    df = pd.read_csv(data_path)
    row_length = len(df)
    column_length = df.columns.size
    if mothed == 'MACD':
        start = 39
        n_m = 1
    elif mothed == 'Aroon':
        start = 24
        n_m = 1
    elif mothed == 'KDJ':
        start = 20
        n_m = 3

    for i in range(start - T + 2, row_length - T):
        X_data = df.iloc[i:i + T - 1, 0:column_length - 1]
        Y_data = df.iloc[i:i + T - 1, column_length - 1]
        label_data = df.iloc[i + T, column_length - 1]
        if mothed == 'MACD':
            trend_data = df.iloc[i + T - 1 - 40:i + T - 1, column_length - 1]
        elif mothed == 'Aroon':
            trend_data = df.iloc[i + T - 1 - 25:i + T - 1, column_length - 1]
        elif mothed == 'KDJ':
            trend_data = df.loc[i + T - 1 - 21:i + T - 1, ['High', 'Low', 'Close']]

        input_X.append(np.array(X_data))
        input_Y.append(np.array(Y_data))
        label_Y.append(np.array(label_data))
        trend_Y.append(np.array(trend_data))

    input_X = np.array(input_X).reshape(-1, T - 1, n)
    input_Y = np.array(input_Y).reshape(-1, T - 1, 1)
    label_Y = np.array(label_Y).reshape(-1, 1)
    trend_Y = np.array(trend_Y).reshape(-1, start + 1, n_m)
    return input_X, input_Y, label_Y, trend_Y


def write_cache(fname,input_X, input_Y, label_Y,trend_Y):
    h5 = h5py.File(fname,'w')
    #h5.create_dataset('num',data=len(input_X))
    h5.create_dataset('input_X',data=input_X)
    h5.create_dataset('input_Y',data=input_Y)
    h5.create_dataset('label_Y',data=label_Y)
    h5.create_dataset('trend_Y', data=trend_Y)
    h5.close()

def read_cache(fname):
    f = h5py.File(fname,'r')
    #num = int(f['num'].value)
    input_X = f['input_X'].value
    input_Y = f['input_Y'].value
    label_Y = f['label_Y'].value
    trend_Y = f['trend_Y'].value
    f.close()
    return input_X,input_Y,label_Y,trend_Y


if CACHEDATA and os.path.isdir(cache_path) is False:
    os.mkdir(cache_path)
fname = os.path.join(cache_path,'nasdaq_T{}.h5'.format(T))
if os.path.exists(fname) and CACHEDATA:
    input_X, input_Y, label_Y,trend_Y = read_cache(fname)
    print('load %s successfully' % fname)
else:
    input_X, input_Y, label_Y,trend_Y = get_data(data_path,T,mothed)
    if CACHEDATA:
        write_cache(fname,input_X, input_Y, label_Y,trend_Y)


input_X_train, input_X_test, input_Y_train,input_Y_test,label_Y_train,label_Y_test,trend_Y_train,trend_Y_test = train_test_split(input_X,input_Y,label_Y,trend_Y, test_size=0.3, random_state=0)
print('input_X_train shape:',input_X_train.shape)
print('input_X_test shape:', input_X_test.shape)
print('input_Y_train shape:', input_Y_train.shape)
print('input_Y_test shape:',input_Y_test.shape)
print('label_Y_train shape:', label_Y_train.shape)
print('label_Y_test shape:', label_Y_test.shape)
print('trend_Y_train shape:', trend_Y_train.shape)
print('trend_Y_test shape:', trend_Y_test.shape)

########################模型构建#######################
en_densor_We = Dense(T)
en_LSTM_cell = LSTM(n_h,return_state=True)
de_LSTM_cell = LSTM(p,return_state=True)
de_densor_We = Dense(m)
LSTM_cell = LSTM(p,return_state=True)

###特征维度注意力机制###
def one_encoder_attention_step(h_prev,s_prev,X):
    '''
    :param h_prev: previous hidden state
    :param s_prev: previous cell state
    :param X: (T,n),n is length of input series at time t,T is length of time series
    :return: x_t's attention weights,total n numbers,sum these are 1
    '''
    concat = Concatenate()([h_prev,s_prev])  #(none,1,2m)
    result1 = en_densor_We(concat)   #(none,1,T)
    result1 = RepeatVector(X.shape[2],)(result1)  #(none,n,T)
    X_temp = My_Transpose(axis=(0,2,1))(X) #X_temp(None,n,T)
    result2 = My_Dot(T)(X_temp)  # (none,n,T)  Ue(T,T)
    result3 = Add()([result1,result2])  #(none,n,T)
    result4 = Activation(activation='tanh')(result3)  #(none,n,T)
    result5 = My_Dot(1)(result4)
    result5 = My_Transpose(axis=(0,2,1))(result5)
    alphas = Activation(activation='softmax')(result5)

    return alphas

def encoder_attention(T,X,y_prev,s0,h0):

    s = s0
    h = h0
    print('s:', s)
    #initialize empty list of outputs
    attention_weight_t = None
    for t in range(T):
        print('X:', X)
        context = one_encoder_attention_step(h,s,X)  #(none,1,n)
        print('context:',context)
        x = Lambda(lambda x: X[:,t,:])(X)
        x = Reshape((1,n))(x)
        print('x:',x)
        h, _, s = en_LSTM_cell(x, initial_state=[h, s])
        if t!=0:
            print('attention_weight_t:',attention_weight_t)
            attention_weight_t= Merge(mode='concat', concat_axis=1)([attention_weight_t,context])
            print('hello')
        else:
            attention_weight_t = context
        print('h:', h)
        print('_:', _)
        print('s:', s)
        print('t', t)
        # break
    # print('@@@@@attention_weight_t:', attention_weight_t)
    X_1 = Multiply()([attention_weight_t,X])
    X_ = Concatenate(axis=2)([y_prev,X_1])
    print('return X:',X_)
    return X_

###时间维度趋势注意力机制###
def one_decoder_attention_step(h_de_prev,s_de_prev,h_en_all,T):
    '''
    :param h_prev: previous hidden state
    :param s_prev: previous cell state
    :param h_en_all: (None,T,m),n is length of input series at time t,T is length of time series
    :return: x_t's attention weights,total n numbers,sum these are 1
    '''
    print('h_en_all:',h_en_all)
    concat = Concatenate()([h_de_prev,s_de_prev])  #(None,1,2p)
    result1 = de_densor_We(concat)   #(None,1,m)
    result1 = RepeatVector(T)(result1)  #(None,T,m)
    result2 = My_Dot(m)(h_en_all)
    print('result2:',result2)
    print('result1:',result1)
    result3 = Add()([result1,result2])  #(None,T,m)
    result4 = Activation(activation='tanh')(result3)  #(None,T,m)
    result5 = My_Dot(1)(result4)

    beta = Activation(activation='softmax')(result5)
    context = Dot(axes = 1)([beta,h_en_all])  #(1,m)
    return context

def decoder_attention(Z1,Z2,Z3,T,h_en_all,Y,s0,h0):
    s = s0
    h = h0
    for t in range(T-1):
        y_prev = Lambda(lambda y_prev: Y[:, t, :])(Y)
        y_prev = Reshape((1, 1))(y_prev)   #(None,1,1)
        print('y_prev:',y_prev)
        context = one_decoder_attention_step(h,s,h_en_all,T)  #(None,1,20)
        context1 = Multiply()([Z1,context])
        context2 = Multiply()([Z2,context])
        context3 = Multiply()([Z3,context])
        y_prev = Concatenate(axis=2)([context1,context2,context3])   #(None,1,21)
        # print('gerald!!!!!!',context)
        print('y_prev:',y_prev)
        y_prev = Dense(1)(y_prev)       #(None,1,1)
        print('y_prev:',y_prev)
        h, _, s = de_LSTM_cell(y_prev, initial_state=[h, s])
        print('h:', h)
        print('_:', _)
        print('s:', s)

    context = one_decoder_attention_step(h, s, h_en_all,T)
    return h,context

###输入及初始化###
X = Input(shape=(T-1,n))   #输入时间序列数据
s0 = Input(shape=(n_s,))  #initialize the first cell state
h0 = Input(shape=(n_h,))   #initialize the first hidden state
h_de0 = Input(shape=(n_hde0,))
s_de0 = Input(shape=(n_sde0,))
Y = Input(shape=(T-1,1))
###特征维度注意力机制###
X_ = encoder_attention(T-1,X,Y,s0,h0)
print('X_:',X_)
X_ = Reshape((T-1,n+1))(X_)
print('X_:',X_)
h_en_all = LSTM(m,return_sequences=True)(X_)
h_en_all = Reshape((T-1,-1))(h_en_all)
print('h_en_all:',h_en_all)
###时间维度趋势注意力机制###
Z1 = Input(shape=(1,m))
Z2 = Input(shape=(1,m))
Z3 = Input(shape=(1,m))
h,context = decoder_attention(Z1,Z2,Z3,T-1,h_en_all,Y,s_de0,h_de0)
###解码器LSTM###
h = Reshape((1,p))(h)
concat = Concatenate(axis=2)([h,context])
concat = Reshape((-1,))(concat)
print('concat:',concat)
###全连接输出层###
result = Dense(p)(concat)
print('result:',result)
output = Dense(1)(result)

s0_train = h0_train = np.zeros((input_X_train.shape[0],m))
h_de0_train = s_de0_train =np.zeros((input_X_train.shape[0],p))
model = Model(inputs=[Z1,Z2,Z3,X,Y,s0,h0,s_de0,h_de0],outputs=output)
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
model.summary()
# early_stopping = EarlyStopping(monitor='val_mean_squared_error',patience=20,mode='min')
# model_checkpoint = ModelCheckpoint(fname_model,monitor='val_mean_squared_error',verbose=0)

########################模型训练#######################
sample_num = input_X_train.shape[0]
#####获取趋势项#####
re_input_Z_1,re_input_Z_2,re_input_Z_3 = get_trend(trend_Y_train,m,sample_num,mothed=mothed)
# re_input_Z_1 = re_input_Z_1.reshape(1529,1,m)
# re_input_Z_2 = re_input_Z_2.reshape(1529,1,m)
# re_input_Z_3 = re_input_Z_3.reshape(1529,1,m)
model.fit([re_input_Z_1,re_input_Z_2,re_input_Z_3,input_X_train,input_Y_train,s0_train,h0_train,s_de0_train,h_de0_train],
          label_Y_train,epochs=epochs,batch_size=batch_size,validation_split=0.2)

########################模型验证#######################
s0_test = h0_test = np.zeros((input_X_test.shape[0],m))
h_de0_test = s_de0_test =np.zeros((input_X_test.shape[0],p))
print(input_X_test.shape)
test_sample_num = input_X_train.shape[0]
#####获取趋势项#####
re_input_Z_1t,re_input_Z_2t,re_input_Z_3t = get_trend(trend_Y_test,m,test_sample_num,mothed=mothed)

score = model.evaluate([re_input_Z_1t,re_input_Z_2t,re_input_Z_3t,input_X_test,input_Y_test,s0_test,h0_test,s_de0_test,h_de0_test],
                       label_Y_test,batch_size=input_X_test.shape[0],verbose=1)
print('loss:',score[0])
print('mse:',score[1])
Y_pred = model.predict([re_input_Z_1t,re_input_Z_2t,re_input_Z_3t,input_X_test,input_Y_test,s0_test,h0_test,s_de0_test,h_de0_test],
                       label_Y_test,batch_size=input_X_test.shape[0],verbose=1)

#####模型评价#####
import matplotlib.pyplot as plt
def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    # ax = fig.add_subplot(111)
    plt.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def evaluation_predicted(predicted_data, true_data):
    y_true, y_pred = np.array(true_data), np.array(predicted_data)
    MAPE = np.mean(np.abs((y_true - y_pred) * 100/ y_true))
    MAE = np.mean(np.abs(y_true - y_pred))
    RMSE = np.sqrt(((y_pred - y_true) ** 2).mean())

    return MAPE,MAE,RMSE
plot_results(Y_pred[:,0],input_Y_test[:,0])
MAE,MAPE,RMSE = evaluation_predicted(Y_pred[:,0],input_Y_test[:,0])
print(MAE,MAPE,RMSE)


