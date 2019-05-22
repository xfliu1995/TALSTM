#!/usr/bin/env python
# -*- coding: utf-8 -*-
# #######趋势判断函数######
import numpy as np
import pandas as pd

def trend_MACD(x):
    n = len(x)
    EMA_12= x.ewm(span=12).mean()
    EMA_26 = x.ewm(span=26).mean()
    DIF = EMA_12 - EMA_26
    DEA = DIF.ewm(span=9).mean()
    MACD = DIF - DEA
    if (MACD[n-1]>0)&(MACD[n-2]>0)&(MACD[n-3]>0)&(MACD[n-4]>0)&(MACD[n-5]>0)&(MACD[n-6]>0)&(MACD[n-7]>0)&(MACD[n-8]>0) \
        &(MACD[n-9]>0)&(MACD[n-10]>0)&(np.abs(MACD[n-1])-np.abs(MACD[n-2])>0)&(np.abs(MACD[n-2])-np.abs(MACD[n-3])>0) \
        &(np.abs(MACD[n-3])-np.abs(MACD[n-4])>0)&(np.abs(MACD[n-4])-np.abs(MACD[n-5])>0)&(np.abs(MACD[n-5])-np.abs(MACD[n-6])>0) \
        & (np.abs(MACD[n-6])-np.abs(MACD[n-7])>0)&(np.abs(MACD[n-7])-np.abs(MACD[n-8])>0)&(np.abs(MACD[n-8])-np.abs(MACD[n-9])>0)&(np.abs(MACD[n-9])-np.abs(MACD[n-10])>0):
        Z = pd.Series([1,0,0])
    elif (MACD[n-1]<0)&(MACD[n-2]<0)&(MACD[n-3]<0)&(MACD[n-4]<0)&(MACD[n-5]<0)&(MACD[n-6]<0)&(MACD[n-7]<0)&(MACD[n-8]<0) \
        &(MACD[n-9]<0)&(MACD[n-10]<0)&(np.abs(MACD[n-1])-np.abs(MACD[n-2])>0)&(np.abs(MACD[n-2])-np.abs(MACD[n-3])>0) \
        &(np.abs(MACD[n-3])-np.abs(MACD[n-4])>0)&(np.abs(MACD[n-4])-np.abs(MACD[n-5])>0)&(np.abs(MACD[n-5])-np.abs(MACD[n-6])>0) \
        & (np.abs(MACD[n-6])-np.abs(MACD[n-7])>0)&(np.abs(MACD[n-7])-np.abs(MACD[n-8])>0)&(np.abs(MACD[n-8])-np.abs(MACD[n-9])>0)&(np.abs(MACD[n-9])-np.abs(MACD[n-10])>0):
        Z = pd.Series([0,1,0])
    else:
        Z = pd.Series([0, 0, 1])
    return Z

def trend_Aroon(x):
    n = len(x)
    x = np.array(x)
    n_max = np.where(x==x.max())[0][0]
    n_min = np.where(x==x.min())[0][0]
    Aroon_up = np.abs(n-(n-1-n_max))/n*100
    Aroon_dn = np.abs(n-(n-1-n_min))/n*100
    Aroon_osc = Aroon_up - Aroon_dn
    if (Aroon_up>70)&(Aroon_osc>0):
        Z = pd.Series([1,0,0])
    elif (Aroon_dn>70)&(Aroon_osc<0):
        Z = pd.Series([0,1,0])
    else:
        Z = pd.Series([0, 0, 1])
    return Z

def trend_KDJ(x):
    n = len(x)
    x = pd.DataFrame(x)
    x.columns =['high','low','close']
    low_list = x['low'].rolling(9).min()
    low_list.fillna(value=x['low'].expanding().min(),inplace=True)
    high_list = x['high'].rolling(9).min()
    high_list.fillna(value=x['high'].expanding().min(),inplace=True)
    rsv = (x['close']-low_list)/(high_list-low_list)*100
    KDJ_K = rsv.ewm(span=2).mean()
    KDJ_D = KDJ_K.ewm(span=2).mean()
    KDJ_J = 3*KDJ_K -2*KDJ_D

    if (KDJ_K[n-1]-KDJ_K[n-2]>0)&(KDJ_K[n-2]-KDJ_K[n-3]>0)&(KDJ_K[n-3]-KDJ_K[n-4]>0)&(KDJ_K[n-4]-KDJ_K[n-5]>0)&(KDJ_K[n-5]-KDJ_K[n-6]>0)&(KDJ_K[n-6]-KDJ_K[n-7]>0) \
        & (KDJ_K[n-7]-KDJ_K[n -8]>0)& (KDJ_K[n-8]-KDJ_K[n -9]>0)& (KDJ_K[n-9]-KDJ_K[n -10]>0) &(KDJ_D[n-1]-KDJ_D[n-2]>0)&(KDJ_D[n-2]-KDJ_D[n-3]>0)&(KDJ_D[n-3]-KDJ_D[n-4]>0)&(KDJ_D[n-4]-KDJ_D[n-5]>0)&(KDJ_D[n-5]-KDJ_D[n-6]>0)&(KDJ_D[n-6]-KDJ_D[n-7]>0) \
        & (KDJ_D[n-7]-KDJ_D[n -8]>0)& (KDJ_D[n-8]-KDJ_D[n -9]>0)& (KDJ_D[n-9]-KDJ_D[n -10]>0) &(KDJ_K[n -10]-KDJ_D[n-10]<0)&(KDJ_K[n -1]-KDJ_D[n-1]>0) :
        Z = np.array([1,0,0])
    elif (KDJ_K[n-1]-KDJ_K[n-2]<0)&(KDJ_K[n-2]-KDJ_K[n-3]<0)&(KDJ_K[n-3]-KDJ_K[n-4]<0)&(KDJ_K[n-4]-KDJ_K[n-5]<0)&(KDJ_K[n-5]-KDJ_K[n-6]<0)&(KDJ_K[n-6]-KDJ_K[n-7]<0) \
        & (KDJ_K[n-7]-KDJ_K[n -8]<0)& (KDJ_K[n-8]-KDJ_K[n -9]<0)& (KDJ_K[n-9]-KDJ_K[n -10]<0) &(KDJ_D[n-1]-KDJ_D[n-2]>0)&(KDJ_D[n-2]-KDJ_D[n-3]>0)&(KDJ_D[n-3]-KDJ_D[n-4]>0)&(KDJ_D[n-4]-KDJ_D[n-5]>0)&(KDJ_D[n-5]-KDJ_D[n-6]>0)&(KDJ_D[n-6]-KDJ_D[n-7]>0) \
        & (KDJ_D[n-7]-KDJ_D[n -8]>0)& (KDJ_D[n-8]-KDJ_D[n -9]>0)& (KDJ_D[n-9]-KDJ_D[n -10]>0) &(KDJ_K[n -10]-KDJ_D[n-10]>0)&(KDJ_K[n -1]-KDJ_D[n-1]<0) :
        Z = np.array([0,1,0])
    else:
        Z = np.array([0, 0, 1])
    return Z


def get_trend(trend_Y,m,sample_num,mothed='MACD'):

    if mothed=='MACD':
        trend_Y_ = pd.DataFrame(trend_Y[:,:,0])
        Z = trend_Y_.apply(lambda x:trend_MACD(x),axis=1)
    elif mothed=='Aroon':
        trend_Y_ = pd.DataFrame(trend_Y[:,:,0])
        Z = trend_Y_.apply(lambda x: trend_Aroon(x), axis=1)
    elif mothed=='KDJ':
        Z = np.zeros([sample_num,3])
        for i in range(sample_num):
            trend_Y_ = trend_Y[i,:,:]
            Z[i,:] = trend_KDJ(trend_Y_)

    Z_1 = np.array(Z)[:,0]
    Z_2 = np.array(Z)[:,1]
    Z_3 = np.array(Z)[:,2]
    Z_1 = np.tile(Z_1, (m,1)).T.reshape(sample_num,1,m)
    Z_2 = np.tile(Z_2, (m, 1)).T.reshape(sample_num,1,m)
    Z_3 = np.tile(Z_3, (m, 1)).T.reshape(sample_num,1,m)
    return Z_1,Z_2,Z_3


if __name__ == '__main__':
    T = 25  # 时间序列窗口大小
    n = 12  # 输入特征个数
    m = n_h = n_s = 20  # length of hidden state m
    p = n_hde0 = n_sde0 = 30  # p
    path = 'D:/pythonfile/gupiaoyuce/attention-LSTM/data/'
    data_path = path + '000002.csv'
    # cache_path = os.path.join(path, 'cache')
    # fname_model = 'model_T{}.h5'.format(T)
    # CACHEDATA = False
    batch_size = 128
    epochs = 100
    test_split = 0.2
    mothed = 'Aroon'


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

        for i in range(start-T+2, row_length - T):
            X_data = df.iloc[i:i + T - 1, 0:column_length - 1]
            Y_data = df.iloc[i:i + T - 1, column_length - 1]
            label_data = df.iloc[i + T, column_length - 1]
            if mothed == 'MACD':
                trend_data = df.iloc[i + T - 1 - 40:i + T - 1, column_length - 1]
            elif mothed == 'Aroon':
                trend_data = df.iloc[i + T - 1 - 25:i + T - 1, column_length - 1]
            elif mothed == 'KDJ':
                trend_data= df.loc[i + T - 1 - 21:i + T - 1, ['High','Low','Close']]

            input_X.append(np.array(X_data))
            input_Y.append(np.array(Y_data))
            label_Y.append(np.array(label_data))
            trend_Y.append(np.array(trend_data))

        input_X = np.array(input_X).reshape(-1, T - 1, n)
        input_Y = np.array(input_Y).reshape(-1, T - 1, 1)
        label_Y = np.array(label_Y).reshape(-1, 1)
        trend_Y = np.array(trend_Y).reshape(-1, start+1, n_m)
        return input_X, input_Y, label_Y, trend_Y

    input_X,input_Y, label_Y, trend_Y = get_data(data_path, T, mothed)
    Z_1,Z_2,Z_3 = get_trend(trend_Y,m=20,sample_num=input_X.shape[0],mothed=mothed)


