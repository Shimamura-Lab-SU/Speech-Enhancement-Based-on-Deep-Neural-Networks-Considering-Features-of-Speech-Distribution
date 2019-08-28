# SpeechEnhancement with Generalized Gaussian Distribution Model

# A_SpeechClassification.py

音声を読み込み，  
'../01 Data/SEGAN/clean_existence'  
'../01 Data/SEGAN/noisy_existence/'  
に平均・標準偏差・betaのファイルを保存する  


# B_data_beta.py

Aで保存したbetaが入ったcsvファイルを読み込み，  
結合し，10以上の値の場合は10に補正する．  
'pkl'にbeta_Trainning.pklを保存  
