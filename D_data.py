#
#	Data Creator for SEGAN
#
#

from __future__ import absolute_import

import math
import wave
import array
import joblib
import glob
import csv


from numba import jit
import numpy as np
import numpy.random as rd
from scipy.signal import lfilter

from D_STB_settings import parameters

import os

# Low Pass Filter for de-emphasis
@jit
def de_emph(y, preemph=0.95):
    if preemph <= 0:
        return y
    return lfilter(1,[1, -preemph], y)

def data_loader(preemph=0.95):
    """
    Read wav files or Load pkl files including wav information
	"""

	# Parameters 読み取り
    args = parameters()

	##  Pklファイルなし → wav読み込み + Pklファイル作成
    ## -------------------------------------------------
    if not os.access(args.clean_pkl_path + '/speech.pkl', os.F_OK):

        ##  Wav ファイルの読み込み
	    # wavファイルパスの獲得
        cname = glob.glob(args.clean_wav_path + '/*.wav')
        nname = glob.glob(args.noisy_wav_path + '/*.wav')
        l = len(cname)  # ファイル数

        # Clean wav の読み込み
        i = 1
        cdata = []
        for cname_ in cname:
            cfile = wave.open(cname_, 'rb')
            cdata.append(np.frombuffer(cfile.readframes(-1), dtype='int16'))
            cfile.close()

            print(' Load Clean wav... #%d / %d' % (i, l))
            i+=1

        cdata = np.concatenate(cdata, axis=0)                # データのシリアライズ
        cdata = cdata - preemph * np.roll(cdata, 1)         # プリエンファシス
        cdata = cdata.astype(np.float32)                    # データ量圧縮(メモリに余裕があるなら消す)
        L = 256                                             # 256サンプル
        D = len(cdata) // L                                 # 分割
        cdata = cdata[:D * L].reshape(D, L)                 # (1,:) --> (D, 256)

        print(' Clean wav is Loaded !!')

        # Noisy wav の読み込み
        i = 1
        ndata = []
        for nname_ in nname:
            nfile = wave.open(nname_, 'rb')
            ndata.append(np.frombuffer(nfile.readframes(-1), dtype='int16'))
            nfile.close()

            print(' Load Noisy wav... #%d / %d' % (i, l))
            i += 1

        ndata =    np.concatenate(ndata, axis=0)            # データのシリアライズ化
        ndata = ndata - preemph * np.roll(ndata, 1)         # プリエンファシス
        ndata = ndata.astype(np.float32)                    # データ量圧縮(メモリに余裕があるなら消す)
        L = 256                                             # 256サンプル
        D = len(ndata) // L                                 # 分割
        ndata = ndata[:D * L].reshape(D, L)                 # (1,:) --> (D, 256)

        sdata=np.concatenate([cdata,ndata])

        print(' Now Creating Pkl file...')

        ##  Pklファイルの作成
        # クリーン+ノイジーpklの作成
        with open(args.clean_pkl_path + '/speech.pkl', 'wb') as f:
            joblib.dump(sdata, f, protocol=-1,compress=3)

        print(' Pkl file is Created !!')

        #データの形状チェック用
        # with open(args.clean_pkl_path + '/speech.csv', 'w') as f:
        #     writer = csv.writer(f, lineterminator='\n')
        #     writer.writerows(sdata)

	##  Pklファイルあり → ロード
    ## -------------------------------------------------
    else:
        # 音声Pklのロード
        print(' Loading Speech wav...')
        with open(args.clean_pkl_path + '/speech.pkl', 'rb') as f:
            sdata = joblib.load(f)

    #betaの読み込み

    #betaPklのロード
    print(' Loading beta...')
    with open(args.clean_pkl_path + '/beta_Training.pkl', 'rb') as f:
        beta = joblib.load(f)

    return sdata,beta


class create_batch:
    """
    Creating Batch Data
    """

    ## 	パラメータおよびデータの保持
    def __init__(self, speech_data, beta_data, batches):
        print(' Create batch object from waveform...')

        # 正規化
        def normalize(data):
            return (1. / 1024.) * data  # [-1024 ~ 1024] -> [-1 ~ 1]

        # データの整形
        self.speech = np.expand_dims(normalize(speech_data),axis=1)    # (D,256,1) -> (D,1,256)
        #self.beta = np.expand_dims(np.expand_dims(beta_data, axis=1),axis=1)
        self.beta = np.expand_dims(beta_data, axis=1)
        #self.beta = np.round(np.expand_dims(beta_data, axis=1))        #ここ変えた
        #self.beta=np.float32(beta_data)
        # ランダムインデックス生成 (データ取り出し用)
        ind = np.array(range(len(speech_data)-1))
        rd.shuffle(ind)

        # パラメータの読み込み
        self.batch = batches
        self.batch_num = math.ceil(len(speech_data)/batches)         # 1エポックあたりの学習数
        self.rnd = np.r_[ind,ind[:self.batch_num*batches-len(speech_data)+1]] # 足りない分は巻き戻して利用
        self.len = len(speech_data)                                  # データ長
        self.index = 0                                              # 読み込み位置



    ## 	データの取り出し
    def next(self, i):

        # データのインデックス指定
        # 各バッチではじめの256サンプル分のインデックス
        index = self.rnd[ i * self.batch : (i + 1) * self.batch ]


        # データ取り出し
        return self.speech[index],\
                self.beta[index]


class create_batch_test:
    """
    Creating Batch Data
    """

    ## 	パラメータおよびデータの保持
    def __init__(self, speech_data, beta_data, batches):
        print(' Create batch object from waveform...')

        # 正規化
        def normalize(data):
            return (1. / 1024.) * data  # [-1024 ~ 1024] -> [-1 ~ 1]

        # データの整形
        self.speech = np.expand_dims(normalize(speech_data), axis=1)  # (D,256,1) -> (D,1,256)
        # self.beta = np.expand_dims(np.expand_dims(beta_data, axis=1),axis=1)
        self.beta = np.expand_dims(beta_data, axis=1)
        # self.beta=np.float32(beta_data)
        # ランダムインデックス生成 (データ取り出し用)
        ind = np.array(range(len(speech_data) - 1))

        # パラメータの読み込み
        self.batch = batches
        self.batch_num = math.ceil(len(speech_data) / batches)  # 1エポックあたりの学習数
        self.rnd = np.r_[ind, ind[:self.batch_num * batches - len(speech_data) + 1]]  # 足りない分は巻き戻して利用
        self.len = len(speech_data)  # データ長
        self.index = 0  # 読み込み位置

    ## 	データの取り出し
    def next(self, i):
        # データのインデックス指定
        # 各バッチではじめの256サンプル分のインデックス
        index = self.rnd[i * self.batch: (i + 1) * self.batch]

        # データ取り出し
        return self.speech[index], \
               self.beta[index]


def wav_write(filename, x, fs=16000):

    # x = de_emph(x)      # De-emphasis using LPF

    x = x * 1024       # denormalized
    x = x.astype('int16')               # cast to int
    w = wave.Wave_write(filename)
    w.setparams((1,     # channel
                 2,     # byte width
                 fs,    # sampling rate
                 len(x),  # number of frames
                 'NONE',
                 'not compressed' # no compression
    ))
    w.writeframes(array.array('h', x).tobytes())
    w.close()

    return 0
