#音声にラベルを付ける
#speech(noisy)_para_mean.csvは音声(雑音)の平均
#speech(noisy)_sd_mean.csvは音声(雑音)の標準偏差
#speech(noisy)_para_beta.csvは近似的に求めたbeta(必ずしも真値ではない)


import os
import wave
import numpy as np
import scipy as sp
import glob
import csv

from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gennorm

# Low Pass Filter for de-emphasis
@jit
def de_emph(y, preemph=0.95):
    if preemph <= 0:
        return y
    return lfilter(1,[1, -preemph], y)

Shift   = 0.5
N       = 8192
preemph=0.95

#音声ファイルの読み込み
speech_dir='../01 Data/SEGAN/clean_trainset_wav_16k/'
noisy_dir='../01 Data/SEGAN/noisy_trainset_wav_16k/'
speech_para='../01 Data/SEGAN/clean_existence'
noisy_para='../01 Data/SEGAN/noisy_existence/'


csave_para='../Distribution/cleanspeech'
nsave_para='../Distribution/noisyspeech'

#音声データのラベル付け
cname=glob.glob(speech_dir+'*.wav')
cdata=[]
for cname_ in cname:
    cfile=wave.open(cname_,'rb')
    cdata.append(np.frombuffer(cfile.readframes(-1), dtype='int16'))                #cdata(List)に0から11572?までデータが入る
    cfile.close()

cdata = np.concatenate(cdata, axis=0)               # データのシリアライズ
cdata = cdata - preemph * np.roll(cdata, 1)         # プリエンファシス
cdata = cdata.astype(np.float32)                    # データ量圧縮(メモリに余裕があるなら消す)
L = 256                                            # フレーム長の半分(256サンプル)
D = len(cdata) // L                                 # 0.015625s(256サンプル)毎に分割
cdata = cdata[:D * L].reshape(D, L)                 # (1,:) --> (2113036,256)


S=N*Shift
n=0
cmean=[0]*D
csd=[0]*D
cbeta=[0]*D
while n<D:
    cmean[n]=np.mean(cdata[n])                          #fit関数を使わずに平均と標準偏差を求める
    csd[n]=np.std(cdata[n])
    n=n+1

#csvで保存
#クリーン音声の平均
with open(os.path.join(speech_para,'speech_para_mean.csv'),'w')as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(cmean)
#クリーン音声の標準偏差
with open(os.path.join(speech_para, 'speech_para_sd.csv'), 'w')as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(csd)

#GGDの形状パラメータbetaを最尤推定を用い求める
n=0
while n<D:
    #para[n]=gennorm.fit(cdata[n])
    cbeta[n],loc,scale=gennorm.fit(cdata[n],floc=0)
    n=n+1
#クリーン音声のbeta
with open(os.path.join(speech_para, 'speech_para_beta.csv'), 'w')as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(cbeta)




#雑音データのラベル付け
nname = glob.glob(noisy_dir + '*.wav')
ndata = []
for nname_ in nname:
    nfile = wave.open(nname_, 'rb')
    ndata.append(np.frombuffer(nfile.readframes(-1), dtype='int16'))  # cdata(List)に0から11572?までデータが入る
    nfile.close()

ndata = np.concatenate(ndata, axis=0)               # データのシリアライズ
ndata = ndata - preemph * np.roll(ndata, 1)         # プリエンファシス
ndata = ndata.astype(np.float32)                    # データ量圧縮(メモリに余裕があるなら消す)
L = 256                                            # フレーム長の半分(256サンプル)
D = len(ndata) // L                                 # 0.015625s(256サンプル)毎に分割
ndata = ndata[:D * L].reshape(D, L)  # (1,:) --> (2113036, 256)




S = N * Shift
n = 0
nmean = [0] * D
nsd = [0] * D
while n < D:
    nmean[n] = np.mean(ndata[n])  # fit関数を使わずに平均と標準偏差を求める
    nsd[n] = np.std(ndata[n])
    n = n + 1

# csvで保存
#雑音混入音声の平均
with open(os.path.join(noisy_para, 'noisy_para_mean.csv'), 'w')as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(nmean)
#雑音混入音声の標準偏差
with open(os.path.join(noisy_para, 'noisy_para_sd.csv'), 'w')as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(nsd)

#GGDの形状パラメータbetaを最尤推定を用い求める
n = 0
nbeta = [0] * D
while n < D:
    nbeta[n],loc,scale = gennorm.fit(ndata[n],floc=0)
    n = n + 1
#雑音混入音声のbeta
with open(os.path.join(noisy_para, 'noisy_para_beta.csv'), 'w')as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(nbeta)


