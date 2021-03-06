#!/usr/bin/env python
# -*- coding: utf-8 -*-

class settings:

    def __init__(self):

        # General Settings
        self.len                = 2 ** 14           # Input Size (len = 16384)
        self.device_id          = 0                 # GPU ID (init:0)
        self.random_seed        = 0
        self.halfprec           = False              # 16Bit or not

        # Parameters
        self.batch_size 	    = 50                # Batch size
        self.epoch              = 100               # Epoch
        self.learning_rate      = 0.0002            # Learning Rate (Generator)
        self.learning_rate_dis  = 0.0002            # Learning Rate (Discriminator)
        self.learning_rate_map  = 0.001             # Learning Rate (Generator)
        self.weight_decay       = 0.0001
        self.decay_rate1        = 0.9
        self.decay_rate2        = 0.99

        # Retrain
        self.epoch_from         = 0                   # Epoch No. from that Retraining starts (init:0)

        # Save path
        #self.model_save_path    = 'unet_tmp.monitor'          # Network model path
        self.model_save_path = 'results_review/20200107'
        self.model_save_cycle   = 2                 # Epoch cycle for saving model (init:1)

        # Save wav path
        #self.wav_save_path      = ('pred_%d'%(self.epoch))
        #self.wav_save_path      = 'results'
        self.wav_save_path = 'results_review/20200107/results/female'

        # Wave files
        self.clean_train_path   = '../01 Data/SEGAN/clean_trainset_wav_16k'     # Folder containing clean wav (train)
        self.noisy_train_path   = '../01 Data/SEGAN/noisy_trainset_wav_16k'     # Folder containing noisy wav (train)
        self.clean_test_path    = '../01 Data/SEGAN/clean_testset_wav_16k/only2'      # Folder containing clean wav (test)
        self.noisy_test_path    = '../01 Data/SEGAN/noisy_testset_wav_16k/only2'      # Folder containing noisy wav (test)

        # Pkl files for train
        self.train_pkl_path     = 'pkl'             # Folder of pkl files for train
        self.train_pkl_clean    = 'train_clean.pkl' # File name of "Clean" pkl for train
        self.train_pkl_noisy    = 'train_noisy.pkl' # File name of "Noisy" pkl for train

        # Pkl files for test
        self.test_pkl_path      = 'pkl'             # Folder of pkl files for test
        self.test_pkl_clean     = 'test_clean.pkl'  # File name of "Clean" pkl for test
        self.test_pkl_noisy     = 'test_noisy.pkl'  # File name of "Noisy" pkl for test
        self.test_pkl_length    = 'test_length.pkl' # File name of "Length" pkl for test

        self.D_monitor_path = 'D_tmp.monitor'  # モニタ関連保存場所
        self.D_model_save_path = self.D_monitor_path  # パラメータ保存先
        self.D_epoch = 100  # Dのエポックと合わせる
