#
# Parameterset
#

# パラメータを詰め込む用変数
class parameters:

    def __init__(self):

        # General Settings
        self.len            = 2 ** 8           # サンプル数

        # Training Parameter
        self.batch_size 	= 50               # バッチサイズ
        self.epoch          = 50                # エポック (1エポック=1回全データ学習)
        self.learning_rate  = 0.0001            # 学習率
        self.weight_decay 	= 0.0001            # 正則化パラメータ

        # Train Condition
        self.retrain        = False              # 過去に学習したパラメータからの再開 or not
        #self.retrain = True  # 過去に学習したパラメータからの再開 or not
        self.pre_epoch      = 20                 # 再開するエポック数

        # Retrain
        self.epoch_from     = 37

        # test
        self.epoch_test     = 37

        # Save path & Load path
        self.monitor_path = 'SE_tmp.monitor'       # モニタ関連保存場所
        self.model_save_path= self.monitor_path # パラメータ保存先
        self.clean_wav_path = '../01 Data/SEGAN/clean_trainset_wav_16k'  # 学習用クリーンwavのパス
        self.noisy_wav_path = '../01 Data/SEGAN/noisy_trainset_wav_16k'  # 学習用ノイジィwavのパス
        self.clean_pkl_path = 'pkl'             # 学習用クリーンpklのパス
        self.noisy_pkl_path = 'pkl'             # 学習用ノイジィpklのパス

        self.D_monitor_path = 'D_tmp.monitor'  # モニタ関連保存場所
        self.D_model_save_path = self.D_monitor_path  # パラメータ保存先
        self.D_epoch=100
