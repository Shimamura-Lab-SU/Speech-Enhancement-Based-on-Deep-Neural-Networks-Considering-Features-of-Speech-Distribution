#
#   Speech Enhancement based on Deep Auto-Encoder using Fully Convolutional Network
#
#


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

#   Figure 関連
import matplotlib.pyplot as plt

from SE_data import pesq_score
from   C_param_for_DAE import parameters
import C_data as dt



# -------------------------------------------
#   Generator ( Encoder + Decoder )
#   - output estimated clean wav
# -------------------------------------------
def Autoencoder(Noisy):
    """
    Building generator network without random latent variables
        Noisy : (Batch, 1, 256)
        Output : (Batch, 1, 256)
    """

    ##  Sub-functions
    ## ---------------------------------
    # Convolution
    def conv(x, output_ch, karnel=(32,), pad=(15,), stride=(2,), name=None):
        return PF.convolution(x, output_ch, karnel, pad=pad, stride=stride, name=name)

    # deconvolution
    def deconv(x, output_ch, karnel=(32,), pad=(15,), stride=(2,), name=None):
        return PF.deconvolution(x, output_ch, karnel, pad=pad, stride=stride, name=name)

    # Activation Function
    def af(x):
        return PF.prelu(x)

    # Concatenate input and skip-input
    def concat(x, h, axis=1):
        return F.concatenate(x, h, axis=axis)

    ##  Main Processing
    ## ---------------------------------
    with nn.parameter_scope("dae"):
        # Enc : Encoder in Generator
        enc1    = af(conv(Noisy, 16, name="enc1"))   # Input:(256, 1) --> (16, 128) *convolutionの結果は自動的に(フィルタ数, 出力サイズ)にreshapeされる
        enc2    = af(conv(enc1, 32, name="enc2"))    # (16, 128) --> (32, 64)
        enc3    = af(conv(enc2, 32, name="enc3"))    # (32, 64) --> (32, 32)
        enc4    = af(conv(enc3, 64, name="enc4"))    # (32, 32) --> (64, 16)
        enc5    = af(conv(enc4, 64, name="enc5"))    # (64, 16) --> (64, 8)
        enc6    = af(conv(enc5, 128, name="enc6"))   # (64, 8) --> (128, 4)
        enc7    = af(conv(enc6, 256, name="enc7"))   # (128, 4) --> (256, 2)


		# Dec : Decoder in Generator
        # Concatenate skip input for each layer
        dec1    = concat(af(deconv(enc7, 128, name="dec1")), enc6) # (256, 2) --> (128, 4) >> [concat](256, 4)
        dec2    = concat(af(deconv(dec1, 64, name="dec2")), enc5)   # (256, 4) --> (64, 8)
        dec3    = concat(af(deconv(dec2, 64, name="dec3")), enc4)   # (64, 8) --> (64, 16)
        dec4    = concat(af(deconv(dec3, 32, name="dec4")), enc3)   # (64, 16) --> (32, 32)
        dec5    = concat(af(deconv(dec4, 32, name="dec5")), enc2)   # (32, 32) --> (32, 64)
        dec6    = concat(af(deconv(dec5, 16, name="dec6")), enc1)    # (32, 64) --> (16, 128)
        dec7   = deconv(dec6, 1, name="dec11")                     # (16, 128) --> (1, 256)

    return F.tanh(dec7)



# -------------------------------------------
#   Loss funcion
# -------------------------------------------
def Loss_reconstruction(wave_fake, wave_true):
   E_wave = F.mean( F.absolute_error(wave_fake, wave_true) )  	# 再構成性能の向上
   return E_wave

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
    noisy 		= nn.Variable([args.batch_size, 1, 256])  # Input
    clean 		= nn.Variable([args.batch_size, 1, 256])  # Desire
    # Network (DAE)
    aeout 	    = Autoencoder(noisy)                        # Predicted Clean
    loss_dae 	= Loss_reconstruction(aeout, clean)         # Loss function

    ##  Declarate Solver
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       - Step 1. Define Solver
    #           * solver_dae : Adam solver (the argument is learning rate)
    #       - Step 2. Set parameters to update
    #           * nn.get_parameters() : parameters in scope "dae"
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Solver
    solver_dae = S.Adam(args.learning_rate)                 # Adam
    # Set parameter
    with nn.parameter_scope("dae"):
        solver_dae.set_parameters(nn.get_parameters())


    ##  Load data & Create batch
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #       - Step 1. Load all learning data using "data_loader"
    #           * clean_data : clean wave data for learning
    #           * noisy_data : noisy wave data for learning
    #       - Step 2. Divide data into batch segment
    #           * create_batch() makes batches including the set of (clean, noisy) from clean/noisy wave data.
    #       - Step 3. Delete all data by "del"
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    clean_data, noisy_data  = dt.data_loader()              # loading all data as nunpy array
    baches  = dt.create_batch(clean_data, noisy_data, args.batch_size)  # creating batch from data
    del clean_data, noisy_data

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
        with nn.parameter_scope("dae"):
            nn.load_parameters(os.path.join(args.model_save_path, "DAE_param_%06d.h5" % args.pre_epoch))
        start_batch_num = args.pre_epoch        # batch num to start
    else:
        start_batch_num = 0                     # batch num to start

    ##  Others
    # only for plot
    ax  = np.linspace(0, 1, 256)      # create axis object
    fig = plt.figure()                  # open fig object (only for plot)


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
            clean.d, noisy.d = baches.next(j)               # Set input data

            # Updating
            solver_dae.zero_grad()                          # Clear the back-propagation result
            loss_dae.forward(clear_no_need_grad=True)       # Run the network
            loss_dae.backward(8, clear_buffer=True)         # Calculate the back-propagation result
            solver_dae.scale_grad(1/8.)
            solver_dae.weight_decay(args.weight_decay*8)    # Set weight-decay parameter
            solver_dae.update()                             # Update

            # Display
            if (j+1) % 500 == 0:
                # Display
                aeout.forward(clear_buffer =True)
                print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('  Epoch #%d, %d/%d  Loss ::' % (i + 1, j + 1, baches.batch_num))
                print('     Reconstruction Error = %.4f' % loss_dae.d)
                print('  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                # Plot
                plt.cla()                               # clear fig object
                plt.plot(ax, aeout.d[0, 0, :])          # output waveform
                plt.plot(ax, clean.d[0, 0, :], color='crimson') # clean waveform
                plt.show(block=False)                   # update fig
                plt.pause(0.0001)                       # pause for drawing

        # Save parameters in scope "dae" for each batch
        with nn.parameter_scope("dae"):
            nn.save_parameters(os.path.join(args.model_save_path, "DAE_param_%06d.h5" % (i + 1)))


    # *****************************************************
    #       Save
    # *****************************************************
    ## Save parameters in scope "dae"
    with nn.parameter_scope("dae"):
        nn.save_parameters(os.path.join(args.model_save_path, "DAE_param_%06d.h5" % args.epoch))


def test(args):

    # *****************************************************
    #       Settings
    # *****************************************************
    ##  Load data & Create batch
    clean_data, noisy_data = dt.data_loader()
    baches_test = dt.create_batch_test(clean_data, noisy_data, start_time=0, stop_time=1000)
    del clean_data, noisy_data

    ##  Create network
    # Variables
    noisy_t     = nn.Variable(baches_test.noisy.shape)  # Input
    output_t    = Autoencoder(noisy_t)                  # Network & Output

    ##  Load parameters
    with nn.parameter_scope("dae"):
        nn.load_parameters(os.path.join(args.model_save_path, "DAE_param_%06d.h5" % args.epoch))

    ##  Validation
    noisy_t.d = baches_test.noisy

    # *****************************************************
    #       Test
    # *****************************************************
    ##  Run (if you wanna run the network, use ".forward()")
    output_t.forward()

    # *****************************************************
    #       Save as wavefile
    # *****************************************************
    ##  Create wav files
    clean = baches_test.clean.flatten()
    output = output_t.d.flatten()
    dt.wav_write('results/clean_C.wav', baches_test.clean.flatten(), fs=16000)
    dt.wav_write('results/input_C.wav', baches_test.noisy.flatten(), fs=16000)
    dt.wav_write('results/output_C.wav', output_t.d.flatten(), fs=16000)

    # *****************************************************
    #       Plot
    # *****************************************************
    ##  Plot
    fig = plt.figure()                      # create fig object
    plt.clf()                               # clear fig object
    ax = np.linspace(0, 1, len(output))     # axis
    plt.plot(ax, output)                    # output waveform
    plt.plot(ax, clean, color='crimson')    # clean waveform
    plt.savefig(os.path.join(args.model_save_path, "figs/test_%06d.png" % args.epoch))# save fig to png



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
    #train(args)

    # Test
    test(args)
    pesq_score('results/clean_C.wav', 'results/output_C.wav')