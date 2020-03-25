# Speech Enhancement with Generalized Gaussian Distribution Model

##  Requrement

### Python

  - Python 3.6
  - CUDA 10.1 & CuDNN 7.6
    + Please choose the appropriate CUDA and CuDNN version to match your [NNabla version](https://github.com/sony/nnabla/releases) 

### Packages

Please install the following packages with pip.
(If necessary, install latest pip first.)

  - nnabla  (over v1.1)
  - nnabla-ext-cuda  (over v1.1)
  - scipy 
  - numba  
  - joblib  
  - pyQT5  
  - pyqtgraph  (after installing pyQT5)
  - pypesq (see ["install with pip"](https://github.com/ludlows/python-pesq#install-with-pip) in offical site)
  
  
# Program

この順で実行する必要あり

# A_SpeechClassification.py

音声からbetaを推定し(最尤推定)，csvファイルで保存するプログラム

音声を読み込み，  
'../01 Data/SEGAN/clean_existence'  
'../01 Data/SEGAN/noisy_existence/'  
に平均・標準偏差・betaのファイルを保存する
(平均・標準偏差は使わないので省略してOK)


# B_data_beta.py

クリーンbetaとノイジーbetaの結合・補正・pkl化

Aで保存したbetaが入ったcsvファイルを読み込み，  
結合・転置し，10以上の値の場合は10に補正する．  
'pkl'にbeta_Training.pklを保存  

# D_SpeechToBeta.py, D_STB_settings.py, D_data.py

音声→betaになるように学習するプログラム

'pkl'の中からspeech.pkl(音声+雑音,なければ作成)とbeta_Training.pkl(Bで作ったファイル)を読み込む  
学習しパラメータを'D_tmp.monitor'に保存
出力は行わず，学習のみを行う

# prop.py, prop_param.py, prop_data.py

Wave-U-Netをもとにした音声強調を行う

'pkl'の中からclean.pkl,noisy.pkl(なければ作成)とDのパラメータを'D_tmp.monitor'から読み込む  
Dのパラメータは更新せずに音声強調を行う
出力音声は'results'の中に保存される



