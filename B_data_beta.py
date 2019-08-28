#betaが入ったcsvファイルを2つ読み込む
#cleanとnoisyのbetaを結合，10より大きい値を10に変更
#pklで保存

import os
import wave
import numpy as np
import scipy as sp
import glob
import csv
import joblib

clean_para='../01 Data/SEGAN/clean_existence'
noisy_para='../01 Data/SEGAN/noisy_existence'
save_para='pkl'

#csvファイルの読み込み

with open(os.path.join(clean_para, 'speech_para_beta.csv')) as f:
    reader = csv.reader(f)
    clean = [row for row in reader]

with open(os.path.join(noisy_para, 'noisy_para_beta.csv')) as f:
    reader = csv.reader(f)
    noisy = [row for row in reader]

t_data=540938091
D=t_data//256

#転置し、betaが10を超える場合は10に変更、結合する

beta=[[0] for j in range(2*D)]
n=0
while n<D:
    if float(clean[0][n])>10:
        beta[n]=10
    else:
        beta[n]=float(clean[0][n])
    n=n+1

m=0
while n<2*D:
    if float(noisy[0][m])>10:
        beta[n]=10
    else:
        beta[n]=float(noisy[0][m])
    n=n+1
    m=m+1


#pklファイルで保存

with open(os.path.join(save_para,'beta_Training.pkl'), 'wb') as f:
        joblib.dump(beta, f, protocol=-1,compress=3)

print(' Pkl file is Created !!')

# with open(os.path.join(save_para, 'beta.csv'), 'w')as f:
#     writer = csv.writer(f, lineterminator='\n')
#     writer.writerow(beta)