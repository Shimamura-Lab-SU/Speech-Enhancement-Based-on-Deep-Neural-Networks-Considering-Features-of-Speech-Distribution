#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from six.moves import range

import os
import time
import numpy as np

# NNabla
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.initializer as I
from nnabla.ext_utils import get_extension_context

# Display
import pyqtgraph as pg
import pyqtgraph.exporters as pgex

# Sound
from scipy.io import wavfile
from pesq import pesq

# Original Functions
from prop_param import settings
import prop_data as dt


# -------------------------------------------
#   PESQ
# -------------------------------------------
## Display progress in console
def pesq_score(clean_wavs, reconst_wavs, band='wb'):
    scores = []

    print('PESQ Calculation...')
    for i, (clean_, reconst_) in enumerate(zip(clean_wavs, reconst_wavs)):
        rate, ref = wavfile.read(clean_)
        rate, deg = wavfile.read(reconst_)
        score = pesq(rate, ref, deg, band)
        scores.append(score)
        print('Score : {0} ... {1}/{2}'.format(score, i, len(clean_wavs)))

    score = np.average(np.array(scores))

    print('  ---------------------------------------------------')
    print('  Average PESQ score = {0}'.format(score))
    print('  ---------------------------------------------------')

    return 0


## Display
class display:
    # Remaining Time Estimation
    class time_estimation:

        def __init__(self, epoch_from, epoch, batch_num):
            self.start = time.time()
            self.epoch = epoch
            self.epoch_from = epoch_from
            self.batch = batch_num
            self.all = batch_num * (epoch - epoch_from)

        def __call__(self, epoch_num, batch_num):
            elapse = time.time() - self.start
            amount = (batch_num + 1) + (epoch_num - self.epoch_from) * self.batch
            remain = elapse / amount * (self.all - amount)

            hours, mins = divmod(elapse, 3600)
            mins, sec = divmod(mins, 60)
            hours_e, mins_e = divmod(remain, 3600)
            mins_e, sec_e = divmod(mins_e, 60)

            elapse_time = [int(hours), int(mins), int(sec)]
            remain_time = [int(hours_e), int(mins_e), int(sec_e)]

            return elapse_time, remain_time

    def __init__(self, epoch_from, epoch, batch_num):
        self.tm = self.time_estimation(epoch_from, epoch, batch_num)
        self.batch = batch_num

    def __call__(self, epoch, trial, losses):
        elapse_time, remain_time = self.tm(epoch, trial)
        print('  ---------------------------------------------------')
        print('  [ Epoch  # {0},    Trials  # {1}/{2} ]'.format(epoch + 1, trial + 1, self.batch))
        print('    +  Mean Squared Loss        = {:.4f}'.format(losses))
        print('    -------------------------')
        print('    +  Elapsed Time            : {0[0]:3d}h {0[1]:02d}m {0[2]:02d}s'.format(elapse_time))
        print('    +  Expected Remaining Time : {0[0]:3d}h {0[1]:02d}m {0[2]:02d}s'.format(remain_time))
        print('  ---------------------------------------------------')


## Create figure object and plot
class figout:
    def __init__(self):
        ## Create Graphic Window
        self.win = pg.GraphicsWindow(title="")
        self.win.resize(800, 600)
        self.win.setWindowTitle('pyqtgraph example: Plotting')
        self.win.setBackground("#FFFFFFFF")
        pg.setConfigOptions(antialias=True)  # Anti-Aliasing for clear plotting

        ## Graph Layout
        #   1st Col: Speech Waveform
        self.p1 = self.win.addPlot(title="Source 1 Waveform")
        self.p1.addLegend()
        self.c11 = self.p1.plot(pen=(255, 0, 0, 255), name="In")
        self.c12 = self.p1.plot(pen=(0, 255, 0, 150), name="Out1")
        self.c13 = self.p1.plot(pen=(0, 0, 255, 90), name="Clean")
        self.win.nextRow()
        self.p2 = self.win.addPlot(title="Source 2 Waveform")
        self.p2.addLegend()
        self.c21 = self.p2.plot(pen=(255, 0, 0, 255), name="In")
        self.c22 = self.p2.plot(pen=(0, 255, 0, 150), name="Out2")
        self.c23 = self.p2.plot(pen=(0, 0, 255, 90), name="Clean")
        self.win.nextRow()
        #   2st Col-1: Loss
        self.p3 = self.win.addPlot(title="Loss")
        self.p3.addLegend()
        self.c31 = self.p3.plot(pen=(255, 0, 0, 255), name="losses")
        self.win.nextRow()

    def waveform_1(self, noisy, target, clean, stride=10):
        self.c11.setData(noisy[0:-1:stride])
        self.c12.setData(target[0:-1:stride])
        self.c13.setData(clean[0:-1:stride])

    def waveform_2(self, noisy, target, clean, stride=10):
        self.c21.setData(noisy[0:-1:stride])
        self.c22.setData(target[0:-1:stride])
        self.c23.setData(clean[0:-1:stride])

    def loss(self, losses, stride=1, log_scale=True):
        if log_scale:
            self.p3.setLogMode(x=False, y=True)
        self.c31.setData(losses[0:-1:stride])


### -------------------------------------------
###   Generator ( Encoder + Decoder )
###   - output estimated clean wav
### -------------------------------------------
def Wave_U_Net(Noisy):
    ds_outputs = list()
    num_initial_filters = 24
    num_layers = 12
    filter_size = 15
    merge_filter_size = 5
    b = I.ConstantInitializer()
    w = I.NormalInitializer(sigma=0.02)

    ##     Sub-functions
    ## ---------------------------------

    #   Convolution
    def conv(x, output_ch, karnel=(15,), pad=(7,), stride=(1,), name=None):
        return PF.convolution(x, output_ch, karnel, pad=pad, stride=stride, w_init=w, b_init=b, name=name)

    #   Activation Function
    def af(x, alpha=0.2):
        return F.leaky_relu(x, alpha)

    #
    def crop_and_concat(x1, x2):

        def crop(tensor, target_times):
            shape = tensor.shape[2]
            diff = shape - target_times
            if diff == 0:
                return tensor
            crop_start = diff // 2
            crop_end = diff - crop_start
            return F.slice(tensor, start=(0, 0, crop_start), stop=(tensor.shape[0], tensor.shape[1], shape - crop_end),
                           step=(1, 1, 1))

        x1 = crop(x1, x2.shape[2])
        return F.concatenate(x1, x2, axis=1)

    def downsampling_block(x, i):
        with nn.parameter_scope(('ds_block-%2d' % i)):
            ds = af(conv(x, (num_initial_filters + num_initial_filters * i), (filter_size,), (7,), name='conv'))
            ds_slice = F.slice(ds, start=(0, 0, 0), stop=ds.shape, step=(1, 1, 2))  # Decimate by factor of 2
            # ds_slice = F.average_pooling(ds, kernel=(1, 1,), stride=(1, 2,), pad=(0, 0,))
            return ds, ds_slice

    def upsampling_block(x, i):

        with nn.parameter_scope(('us_block-%2d' % i)):
            up = F.unpooling(af(x), (2,))
            cac_x = crop_and_concat(ds_outputs[-i - 1], up)
            us = af(conv(cac_x, num_initial_filters + num_initial_filters * (num_layers - i - 1), (merge_filter_size,),
                         (2,), name='conv'))
            return us

    with nn.parameter_scope('Wave-U-Net'):
        current_layer = Noisy
        ## downsampling block
        for i in range(num_layers):
            ds, current_layer = downsampling_block(current_layer, i)
            ds_outputs.append(ds)
        ## latent variable
        with nn.parameter_scope('latent_variable'):
            current_layer = af(conv(current_layer, num_initial_filters + num_initial_filters * num_layers))
        ## upsampling block
        for i in range(num_layers):
            current_layer = upsampling_block(current_layer, i)

        current_layer = crop_and_concat(Noisy, current_layer)

        ## output layer
        target_1 = F.tanh(conv(current_layer, 1, (1,), (0,), name='target_1'))
        target_2 = F.tanh(conv(current_layer, 1, (1,), (0,), name='target_2'))
        return target_1, target_2

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


def Loss_reconstruction(beta_in,beta_clean):
   B_wave = F.mean( F.absolute_error(beta_in, beta_clean) )
   return 0.001*B_wave                                          #係数が分布推定ネットワークの重みになる

# -------------------------------------------
#   Train processing
# -------------------------------------------
def train(args):
    ##  Sub-functions
    ## ---------------------------------
    ## Save Models
    def save_models(epoch_num, losses):

        # save generator parameter
        with nn.parameter_scope('Wave-U-Net'):
            nn.save_parameters(os.path.join(args.model_save_path, 'param_{:04}.h5'.format(epoch_num + 1)))

        # save results
        np.save(os.path.join(args.model_save_path, 'losses_{:04}.npy'.format(epoch_num + 1)), np.array(losses))

    ## Load Models
    def load_models(epoch_num, gen=True, dis=True):

        # load generator parameter
        with nn.parameter_scope('Wave-U-Net'):
            nn.load_parameters(os.path.join(args.model_save_path, 'param_{:04}.h5'.format(args.epoch_from)))

    ## Update parameters
    class updating:

        def __init__(self):
            self.scale = 8 if args.halfprec else 1

        def __call__(self, solver, loss):
            solver.zero_grad()  # initialize
            loss.forward(clear_no_need_grad=True)  # calculate forward
            loss.backward(self.scale, clear_buffer=True)  # calculate backward
            # solver.scale_grad(1. / self.scale)                # scaling
            solver.update()  # update

    ##  Inital Settings
    ## ---------------------------------

    ##  Create network
    #   Clear
    nn.clear_parameters()
    #   Variables
    noisy = nn.Variable([args.batch_size, 1, 16384], need_grad=False)  # Input
    clean = nn.Variable([args.batch_size, 1, 16384], need_grad=False)  # Desire
    dis_in = nn.Variable([args.batch_size, 1, 256])
    dis_clean = nn.Variable([args.batch_size, 1, 256])
    beta_in = nn.Variable([args.batch_size, 1, 16384 / 256])
    beta_clean = nn.Variable([args.batch_size, 1, 16384 / 256])
    beta_buf_in = np.empty((args.batch_size, 1, 64))
    beta_buf_cl = np.empty((args.batch_size, 1, 64))

    # Build Network
    # K=2, C=1
    target_1, target_2 = Wave_U_Net(noisy)
    output_t = Discriminator(dis_in)
    c_beta = Discriminator(dis_clean)

    # Mean Squared Error
    loss = (F.mean(F.squared_error(clean, target_1)) + F.mean(F.squared_error(noisy - clean, target_2))) / 2\
           +Loss_reconstruction(beta_in,beta_clean)

    # Optimizer: Adam
    solver = S.Adam(args.learning_rate)

    # set parameter
    with nn.parameter_scope('Wave-U-Net'):
        solver.set_parameters(nn.get_parameters())

    ##  Load data & Create batch
    clean_data, noisy_data = dt.data_loader()
    batches = dt.create_batch(clean_data, noisy_data, args.batch_size)
    del clean_data, noisy_data

    ##  Initial settings for sub-functions
    fig = figout()
    disp = display(args.epoch_from, args.epoch, batches.batch_num)
    upd = updating()

    ##  Train
    ##----------------------------------------------------

    ##  Load parameters
    with nn.parameter_scope("stb"):
        nn.load_parameters(os.path.join(args.D_model_save_path, "STB_param_%06d.h5" % args.D_epoch))

    print('== Start Training ==')

    ##  Load "Pre-trained" parameters
    if args.epoch_from > 0:
        print(' Retrain parameter from pre-trained network')
        load_models(args.epoch_from)
        losses = np.load(os.path.join(args.model_save_path, 'losses_{:04}.npy'.format(args.epoch_from)))

        ## Create loss loggers
        point = args.epoch_from * ((batches.batch_num + 1) // 10)
        loss_len = (args.epoch - args.epoch_from) * ((batches.batch_num + 1) // 10)
        losses = np.append(losses, np.zeros(loss_len))
    else:
        losses = []
        ## Create loss loggers
        point = len(losses)
        loss_len = (args.epoch - args.epoch_from) * ((batches.batch_num + 1) // 10)
        losses = np.append(losses, np.zeros(loss_len))

    ##  Training
    for i in range(args.epoch_from, args.epoch):

        print('')
        print(' =========================================================')
        print('  Epoch :: {0}/{1}'.format(i + 1, args.epoch))
        print(' =========================================================')
        print('')

        #  Batch iteration
        for j in range(batches.batch_num):
            print('  Train (Epoch. {0}) - {1}/{2}'.format(i + 1, j + 2, batches.batch_num))

            ##  Batch setting
            clean.d, noisy.d = batches.next(j)

            ##  Updating
            upd(solver, loss)  # update Generator

            for k in range(64):
                dis_in.d = target_1.d[::, ::, 256 * k:256 * (k + 1):]
                dis_clean.d = clean.d[::, ::, 256 * k:256 * (k + 1):]
                output_t.forward()
                beta_buf_in[:, :, k] = output_t.d
                c_beta.forward()
                beta_buf_cl[:, :, k] = c_beta.d

            beta_in.d = beta_buf_in
            beta_clean.d = beta_buf_cl


            ##  Display
            if (j) % 100 == 0:
                # Get result for Display
                target_1.forward(clear_no_need_grad=True)
                target_2.forward(clear_no_need_grad=True)



                # Display text
                disp(i, j, loss.d)

                # Data logger
                losses[point] = loss.d
                point = point + 1

                # Plot
                fig.waveform_1(noisy.d[0, 0, :], target_1.d[0, 0, :], clean.d[0, 0, :])
                fig.waveform_2(noisy.d[0, 0, :], target_2.d[0, 0, :], clean.d[0, 0, :])
                fig.loss(losses[0:point - 1])
                pg.QtGui.QApplication.processEvents()

        ## Save parameters
        if ((i + 1) % args.model_save_cycle) == 0:
            save_models(i, losses)  # save model
            # fig.save(os.path.join(args.model_save_path, 'plot_{:04}.pdf'.format(i + 1))) # save fig
            exporter = pg.exporters.ImageExporter(
                fig.win.scene())  # exportersの直前に pg.QtGui.QApplication.processEvents() を呼ぶ！
            exporter.export(os.path.join(args.model_save_path, 'plot_{:04}.png'.format(i + 1)))  # save fig

    ## Save parameters (Last)
    save_models(args.epoch - 1, losses)
    exporter = pg.exporters.ImageExporter(fig.win.scene())  # exportersの直前に pg.QtGui.QApplication.processEvents() を呼ぶ！
    exporter.export(os.path.join(args.model_save_path, 'plot_{:04}.png'.format(i + 1)))  # save fig


# -------------------------------------------
#   Test processing
# -------------------------------------------
def test(args):
    ##  Load data & Create batch
    clean_data, noisy_data, length_data = dt.data_loader(test=True, need_length=True)
    print(clean_data.shape)
    # Batch
    #  - Proccessing speech interval can be adjusted by "start_frame" and "start_frame".
    #  - "None" -> All speech in test dataset.

    output_ts = []
    bt_idx = 0
    test_batch_size = args.batch_size
    for i in range(clean_data.shape[0] // (test_batch_size * 2)):
        print(i, "/", clean_data.shape[0] // (test_batch_size * 2))
        batches_test = dt.create_batch_test(clean_data[bt_idx:bt_idx + test_batch_size * 2],
                                            noisy_data[bt_idx:bt_idx + test_batch_size * 2], start_frame=0,
                                            stop_frame=test_batch_size * 2)

        ##  Create network
        # Variables
        noisy_t = nn.Variable(batches_test.noisy.shape)  # Input
        # Network (Only Generator)
        output_t, _ = Wave_U_Net(noisy_t)
        ##  Load parameter
        # load generator
        with nn.parameter_scope('Wave-U-Net'):
            nn.load_parameters(os.path.join(args.model_save_path, "param_{:04}.h5".format(args.epoch)))
        ##  Validation
        noisy_t.d = batches_test.noisy
        output_t.forward()

        ##  Create wav files
        output = output_t.d.flatten()
        output_ts.append(output)
        bt_idx += (test_batch_size * 2)

    if (clean_data.shape[0] % (test_batch_size * 2)) != 0:
        last_batch_size_2 = clean_data.shape[0] % (test_batch_size * 2)
        print(last_batch_size_2)
        batches_test = dt.create_batch_test(clean_data[bt_idx:bt_idx + last_batch_size_2],
                                            noisy_data[bt_idx:bt_idx + last_batch_size_2], start_frame=0,
                                            stop_frame=last_batch_size_2)

        ##  Create network
        # Variables
        noisy_t = nn.Variable(batches_test.noisy.shape)  # Input
        # Network (Only Generator)
        ##  Load parameter
        output_t, _ = Wave_U_Net(noisy_t)
        # load generator
        with nn.parameter_scope('Wave-U-Net'):
            nn.load_parameters(os.path.join(args.model_save_path, "param_{:04}.h5".format(args.epoch)))
        ##  Validation
        noisy_t.d = batches_test.noisy
        output_t.forward()

        ##  Create wav files
        output = output_t.d.flatten()
        output_ts.append(output)
        bt_idx += (last_batch_size_2)

    output = output_ts[0]
    for i in range(1, len(output_ts)):
        output = np.concatenate([output, output_ts[i]], axis=0)
    print(len(output))
    output = np.array(output)
    print(output.shape)
    idx_cnt = 0
    for i in range(len(length_data['name'])):
        print("saving", i, length_data['name'][i], "...")
        outwav = output[idx_cnt:idx_cnt + length_data['length'][i]]
        print(outwav.shape)
        idx_cnt += length_data['length'][i]
        print(idx_cnt)
        dt.wav_write((args.wav_save_path + '/' + 'ests_' + os.path.basename(length_data['name'][i])), np.array(outwav),
                     fs=16000)
    print('finish!')


if __name__ == '__main__':

    ## Load settings
    args = settings()

    ## GPU connection
    if args.halfprec:
        # - Float 16-bit precision mode : When GPU memory often gets stack, please use it.
        ctx = get_extension_context('cudnn', device_id=args.device_id, type_config='half')
    else:
        # - Float 32-bit precision mode :
        ctx = get_extension_context('cudnn', device_id=args.device_id)

    ## Training or Prediction
    Train = 0               #Train:1 Test:0
    if Train:
        # Training
        nn.set_default_context(ctx)
        train(args)
    else:
        # Test
        test(args)
        import glob

        clean_wavs = glob.glob(args.clean_test_path + '/*.wav')
        #reconst_wavs = glob.glob('results' + '/*.wav')
        reconst_wavs = glob.glob('results_review/20200107/results/female' + '/*.wav')
        pesq_score(clean_wavs, reconst_wavs)
