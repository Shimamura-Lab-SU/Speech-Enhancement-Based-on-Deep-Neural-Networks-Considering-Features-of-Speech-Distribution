#音声→betaとなるように学習
#STB_settings.py    パラメータ類
#data.py            音声を256サンプル毎に分割・cleanとnoisyを統合・pklファイルで保存
#data_beta.py       betaが入ったcsvファイルを読み込み・cleanとnoisyを統合・大きい値を補正・pklファイルで保存(事前に実行)


from __future__ import absolute_import
from six.moves import range

import os
import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.initializer as I
from nnabla.ext_utils import get_extension_context
import joblib
import csv

#   Figure 関連
import matplotlib.pyplot as plt

from D_STB_settings import parameters
import D_data as dt




# -------------------------------------------
#   Discriminator
# -------------------------------------------
def Discriminator(speech):
    """
    Building discriminator network
        Noisy : (Batch, 1, 256)
        Clean : (Batch, 1, 256)
        Output : (Batch, 1, 256)
    """

    ##  Sub-functions
    ## ---------------------------------
    # Convolution + Batch Normalization
    def n_conv(x, output_ch, karnel=(31,), pad=(15,), stride=(2,), name=None):
        # return PF.batch_normalization(
        #     PF.convolution(x, output_ch, karnel, pad=pad, stride=stride, name=name),
        #     batch_stat=not test,
        #     name=name)
        return PF.convolution(x, output_ch, karnel, pad=pad, stride=stride, name=name)
    # Activation Function
    def af(x):
        return F.leaky_relu(x)

    ##  Main Processing
    ## ---------------------------------
    #Input = F.concatenate(Noisy, Clean, axis=1)
    # Dis : Discriminator
    with nn.parameter_scope("stb"):
        dis1 = af(n_conv(speech, 8, name="dis1"))  # Input:(2, 16384) --> (16, 16384)
        dis2 = af(n_conv(dis1, 16, name="dis2"))  # (16, 16384) --> (32, 8192)
        dis3 = af(n_conv(dis2, 16, name="dis3"))  # (32, 8192) --> (32, 4096)
        dis4 = af(n_conv(dis3, 32, name="dis4"))  # (32, 4096) --> (64, 2048)
        dis5 = af(n_conv(dis4, 32, name="dis5"))  # (64, 2048) --> (64, 1024)
        dis6 = af(n_conv(dis5, 64, name="dis6"))  # (64, 1024) --> (128, 512)
        dis7 = n_conv(dis6, 128, name="dis7")  # (512, 32) --> (1024, 16)
        f = PF.affine(dis7,1)  # (1024, 16) --> (1,)
        #f=F.tanh(dis7)

    return f


# -------------------------------------------
#   Loss funcion (sub functions)
# -------------------------------------------
def SquaredError_Scalor(x, val=1):
    return F.squared_error(x, F.constant(val, x.shape))


# -------------------------------------------
#   Loss funcion
# -------------------------------------------
def Loss_stb(dval_real, dval_fake):
    E_real = F.mean(SquaredError_Scalor(dval_real, val=1))  # real
    E_fake = F.mean(SquaredError_Scalor(dval_fake, val=0))  # fake
    return E_real + E_fake




# -------------------------------------------
#   Train processing
# -------------------------------------------
def train(args):


    # *****************************************************
    #       Settings
    # *****************************************************

    ##  Declarate Network
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       - Step 1. Define Variables
    #           * noisy : container of batch of input values
    #           * clean : container of batch of true values
    #       - Step 2. Define Network
    #           * aeout : output of Network using "Autoencoder"
    #           * loss_dae : loss function
    #       - Step 3. Define Solver
    #           * solver_dae : Adam function
    #       - Step 4. Define Solver
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Variables
    speech 		= nn.Variable([args.batch_size, 1, 256])   # Input
    beta        = nn.Variable([args.batch_size, 1])     # Desire
    # Network (DAE)
    stbout 	    = Discriminator(speech)                     # Predicted Clean
    loss_stb 	= F.mean(F.squared_error(stbout, beta))            # Loss function

    ##  Declarate Solver
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       - Step 1. Define Solver
    #           * solver_dae : Adam solver (the argument is learning rate)
    #       - Step 2. Set parameters to update
    #           * nn.get_parameters() : parameters in scope "dae"
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solver
    solver_stb = S.Adam(args.learning_rate)                 # Adam
    # Set parameter
    with nn.parameter_scope("stb"):
        solver_stb.set_parameters(nn.get_parameters())

    ##  Load data & Create batch
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       - Step 1. Load all learning data using "data_loader"
    #           * clean_data : clean wave data for learning
    #           * noisy_data : noisy wave data for learning
    #       - Step 2. Divide data into batch segment
    #           * create_batch() makes batches including the set of (clean, noisy) from clean/noisy wave data.
    #       - Step 3. Delete all data by "del"
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    speech_data ,beta_data = dt.data_loader()              # loading all data as nunpy array
    baches  = dt.create_batch(speech_data,beta_data, args.batch_size)  # creating batch from data
    del speech_data,beta_data

    ##  Reconstruct parameters
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       If "retrain" is true,
    #           - load trained parameters from "DAE_param_%06d.h5"
    #       Otherwise
    #           - do nothing
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if args.retrain:
        # Reconstruct parameters
        print(' Retrain parameter from past-trained network')
        with nn.parameter_scope("stb"):
            nn.load_parameters(os.path.join(args.model_save_path, "STB_param_%06d.h5" % args.pre_epoch))
        start_batch_num = args.pre_epoch        # batch num to start
    else:
        start_batch_num = 0                     # batch num to start




    # *****************************************************
    #       Training
    # *****************************************************
    print('== Start Training ==')
    for i in range(start_batch_num, args.epoch):

        print('--------------------------------')
        print(' Epoch :: %d/%d' % (i + 1, args.epoch))
        print('--------------------------------')

        #  Batch iteration
        for j in range(baches.batch_num):
            print('  Train (Epoch.%d) - %d/%d' % (i+1, j+1, baches.batch_num))

            # Set input data
            speech.d,beta.d = baches.next(j)               # Set input data
            # Updating
            solver_stb.zero_grad()                          # Clear the back-propagation result
            loss_stb.forward(clear_no_need_grad=True)       # Run the network
            loss_stb.backward(8, clear_buffer=True)         # Calculate the back-propagation result
            solver_stb.scale_grad(1/8.)
            solver_stb.weight_decay(args.weight_decay*8)    # Set weight-decay parameter
            solver_stb.update()                             # Update

            # Display
            if (j+1) % 50 == 0:
                # Display
                stbout.forward(clear_buffer =True)
                print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('  Epoch #%d, %d/%d  Loss ::' % (i + 1, j + 1, baches.batch_num))
                print('     Reconstruction Error = %.4f' % loss_stb.d)
                print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        # Save parameters in scope "dae" for each batch
        with nn.parameter_scope("stb"):
            nn.save_parameters(os.path.join(args.model_save_path, "STB_param_%06d.h5" % (i + 1)))


    # *****************************************************
    #       Save
    # *****************************************************
    ## Save parameters in scope "dae"
    with nn.parameter_scope("stb"):
        nn.save_parameters(os.path.join(args.model_save_path, "STB_param_%06d.h5" % args.epoch))


def test(args):


    ##  Load parameters
    with nn.parameter_scope("stb"):
        nn.load_parameters(os.path.join(args.model_save_path, "STB_param_%06d.h5" % args.epoch))

    ##  Load data & Create batch
    speech_data, beta_data = dt.data_loader()  # loading all data as nunpy array
    baches_test = dt.create_batch_test(speech_data, beta_data, args.batch_size)  # creating batch from data
    del speech_data, beta_data

    # Variables
    speech_t = nn.Variable([args.batch_size, 1, 256])  # Input
    # Network (DAE)
    output_t = Discriminator(speech_t)  # Predicted Clean

    print('== Start Test ==')

    #  Batch iteration
    for j in range(baches_test.batch_num):
        print('  Test  - %d/%d' % ( j + 1, baches_test.batch_num))

        # Set input data
        speech_t.d, _ = baches_test.next(j)  # Set input data
        #speech_t.d=baches_test.speech
        output_t.forward()

    # *****************************************************
    #       Save as pklfile
    # *****************************************************

    output = output_t.d.T             # これだと横に出力される
    with open(args.result_path + '/beta_result.pkl', 'wb') as f:
        joblib.dump(output, f, protocol=-1, compress=3)






if __name__ == '__main__':

    # GPU connection
    ctx = get_extension_context('cudnn', device_id=0, type_config='half')
    nn.set_default_context(ctx)

    # Load parameters
    args = parameters()


    # Training
    #   1. Pre-train for only generator
    #       -- if "pretrain"
    #           - if "retrain"     -> load trained-generator & restart pre-train
    #           - else             -> initialize generator & start pre-train
    #       -- else                -> nothing
    #   2. Train
    #       -- if "retrain"        -> load trianed-generator and trained-discriminator & restart train
    #       -- else                -> start train (* normal case)
    train(args)

    # Test
    #test(args)

