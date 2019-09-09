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
from    SE_param import parameters
import  SE_data as dt



# -------------------------------------------
#   Generator ( Encoder + Decoder )
#   - output estimated clean wav
# -------------------------------------------
def Autoencoder(Noisy):
    """
    Building generator network without random latent variables
        Noisy : (Batch, 1, 16384)
        Output : (Batch, 1, 16384)
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
        enc1 = af(conv(Noisy, 16, name="enc1"))  # Input:(16384, 1) --> (16, 8192) *convolutionの結果は自動的に(フィルタ数, 出力サイズ)にreshapeされる
        enc2 = af(conv(enc1, 32, name="enc2"))  # (16, 8192) --> (32, 4096)
        enc3 = af(conv(enc2, 32, name="enc3"))  # (32, 4096) --> (32, 2048)
        enc4 = af(conv(enc3, 64, name="enc4"))  # (32, 2048) --> (64, 1024)
        enc5 = af(conv(enc4, 64, name="enc5"))  # (64, 1024) --> (64, 512)
        enc6 = af(conv(enc5, 128, name="enc6"))  # (64, 512) --> (128, 256)
        enc7 = af(conv(enc6, 128, name="enc7"))  # (128, 256) --> (128, 128)
        enc8 = af(conv(enc7, 256, name="enc8"))  # (128, 128) --> (256, 64)
        enc9 = af(conv(enc8, 256, name="enc9"))  # (256, 64) --> (256, 32)
        enc10 = af(conv(enc9, 512, name="enc10"))  # (256, 32) --> (512, 16)
        enc11 = af(conv(enc10, 1024, name="enc11"))  # (512, 16) --> (1024, 8)

        # Dec : Decoder in Generator
        # Concatenate skip input for each layer
        dec1 = concat(af(deconv(enc11, 512, name="dec1")), enc10)  # (1024, 8) --> (512, 16) >> [concat](1024, 16)
        dec2 = concat(af(deconv(dec1, 256, name="dec2")), enc9)  # (1024, 16) --> (256, 32)
        dec3 = concat(af(deconv(dec2, 256, name="dec3")), enc8)  # (512, 32) --> (256, 64)
        dec4 = concat(af(deconv(dec3, 128, name="dec4")), enc7)  # (512, 128) --> (128, 256)
        dec5 = concat(af(deconv(dec4, 128, name="dec5")), enc6)  # (512, 128) --> (128, 256)
        dec6 = concat(af(deconv(dec5, 64, name="dec6")), enc5)  # (512, 256) --> (64, 512)
        dec7 = concat(af(deconv(dec6, 64, name="dec7")), enc4)  # (128, 512) --> (64, 1024)
        dec8 = concat(af(deconv(dec7, 32, name="dec8")), enc3)  # (128, 1024) --> (32, 2048)
        dec9 = concat(af(deconv(dec8, 32, name="dec9")), enc2)  # (64, 2048) --> (32, 4096)
        dec10 = concat(af(deconv(dec9, 16, name="dec10")), enc1)  # (32, 4096) --> (16, 8192)
        dec11 = deconv(dec10, 1, name="dec11")  # (32, 8192) --> (1, 16384)

    return F.tanh(dec11)


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
        dis1 = af(n_conv(speech, 8, name="dis1"))
        dis2 = af(n_conv(dis1, 16, name="dis2"))
        dis3 = af(n_conv(dis2, 16, name="dis3"))
        dis4 = af(n_conv(dis3, 32, name="dis4"))
        dis5 = af(n_conv(dis4, 32, name="dis5"))
        dis6 = af(n_conv(dis5, 64, name="dis6"))
        dis7 = n_conv(dis6, 128, name="dis7")
        f = PF.affine(dis7,1)

    return f



# -------------------------------------------
#   Loss funcion
# -------------------------------------------
def Loss_reconstruction(wave_fake, wave_true, beta_in,beta_clean):
   E_wave = F.mean( F.absolute_error(wave_fake, wave_true) )  	# 再構成性能の向上
   B_wave = F.mean( F.absolute_error(beta_in, beta_clean) )
   return E_wave+0.01*B_wave

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
    noisy 		= nn.Variable([args.batch_size, 1, 16384])  # Input
    clean 		= nn.Variable([args.batch_size, 1, 16384])  # Desire
    dis_in     = nn.Variable([args.batch_size, 1, 256])
    dis_clean   = nn.Variable([args.batch_size, 1, 256])
    beta_in = nn.Variable([args.batch_size,1, 16384/256])
    beta_clean = nn.Variable([args.batch_size,1, 16384/256])
    beta_buf_in = np.empty((args.batch_size,1, 64))
    beta_buf_cl = np.empty((args.batch_size,1, 64))

    # Network (DAE)
    aeout 	    = Autoencoder(noisy)                        # Predicted Clean
    output_t = Discriminator(dis_in)
    c_beta      = Discriminator(dis_clean)

    loss_dae 	= Loss_reconstruction(aeout, clean, beta_in,beta_clean) # Loss function



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

    #nn.clear_parameters()
    ##  Load parameters
    with nn.parameter_scope("stb"):
        nn.load_parameters(os.path.join(args.D_model_save_path, "STB_param_%06d.h5" % args.D_epoch))

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
    ax  = np.linspace(0, 1, 16384)      # create axis object
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

            # Calculating beta from segmented signals
            aeout.forward(clear_buffer=True)
            for k in range(64):
                dis_in.d=aeout.d[::,::,256*k:256*(k+1):]
                dis_clean.d=clean.d[::,::,256*k:256*(k+1):]
                output_t.forward()
                beta_buf_in[:,:,k]=output_t.d
                c_beta.forward()
                beta_buf_cl[:,:,k]=c_beta.d

            beta_in.d = beta_buf_in
            beta_clean.d = beta_buf_cl


            # Updating
            solver_dae.zero_grad()                          # Clear the back-propagation result
            loss_dae.forward(clear_no_need_grad=True)       # Run the network
            loss_dae.backward(8, clear_buffer=True)         # Calculate the back-propagation result
            solver_dae.scale_grad(1/8.)
            solver_dae.weight_decay(args.weight_decay*8)    # Set weight-decay parameter
            solver_dae.update()                             # Update




            # Display
            if (j+1) % 50 == 0:
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
    baches_test = dt.create_batch_test(clean_data, noisy_data, start_time=0, stop_time=20)
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
    dt.wav_write('results/clean_SE.wav', baches_test.clean.flatten(), fs=16000)
    dt.wav_write('results/input_SE.wav', baches_test.noisy.flatten(), fs=16000)
    dt.wav_write('results/output_SE.wav', output_t.d.flatten(), fs=16000)

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
    train(args)

    # Test
    #test(args)
    #pesq_score('results/clean_SE.wav','results/output_SE.wav')
