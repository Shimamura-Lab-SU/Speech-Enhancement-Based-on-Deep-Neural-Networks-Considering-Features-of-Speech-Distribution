# SpeechEnhancement with Generalized Gaussian Distribution Model

この順で実行する必要あり

# SpeechClassification.py

音声からbetaを推定し，保存するプログラム

音声を読み込み，  
'../01 Data/SEGAN/clean_existence'  
'../01 Data/SEGAN/noisy_existence/'  
に平均・標準偏差・betaのファイルを保存する  


# data_beta.py

クリーンbetaとノイジーbetaの結合・補正・pkl化

Aで保存したbetaが入ったcsvファイルを読み込み，  
結合し，10以上の値の場合は10に補正する．  
'pkl'にbeta_Trainning.pklを保存  

# D_SpeechToBeta.py, D_STB_settings.py, D_data.py

音声→betaになるように学習するプログラム

[Training]
※SE~を実行する前に学習が必要  
pklの中からspeech.pkl(音声+雑音,なければ作成)とbeta_Training.pkl(Bで作ったファイル)を読み込む  
学習しパラメータをD_tmp.monitorに保存  

# SE_withB.py, SE_data.py, SE_param.py

音声強調部分

pklの中からclean.pkl,noisy.pkl(なければ作成)とDのパラメータを読み込む  
Dのパラメータは更新せずに音声強調を行う
