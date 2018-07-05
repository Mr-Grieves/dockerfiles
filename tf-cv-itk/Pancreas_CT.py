#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:17:13 2018

@author: davood
"""



import numpy as np
#import os
import tensorflow as tf
import tensorlayer as tl
#from os import listdir
#from os.path import isfile, join
import scipy.io as sio
#from skimage import io
#import skimage
import SimpleITK as sitk
#import matplotlib.pyplot as plt
import h5py
#import scipy
#from scipy.spatial import ConvexHull
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.spatial import Delaunay


data_dir=  '/davood_data/'
#thmbs_dir= '/media/davood/dkres/Pancreas_CT/thumbs/'
#mhd_dir=   '/media/davood/dkres/Pancreas_CT/mhd/'


#h5f = h5py.File(data_dir + 'Pancreas_CT.h5','r')
#X= h5f['X_train'][:]
#Y= h5f['Y_train'][:]
#Z= h5f['Z_train'][:]
#h5f.close()
#
#Y[:,:,:,:,1] = np.float32(Y[:,:,:,:,1]>0)
#Y[:,:,:,:,0]= 1-Y[:,:,:,:,1]
#
#X_train= X[:200,:,:,:,:].copy()
#Y_train= Y[:200,:,:,:,:].copy()
#Z_train= Z[:200,:].copy()
#X_test = X[200:,:,:,:,:].copy()
#Y_test = Y[200:,:,:,:,:].copy()
#Z_test = Z[200:,:].copy()
#
#h5f = h5py.File(data_dir + 'Pancreas_CT_train_test.h5','w')
#h5f['X_train']= X_train
#h5f['Y_train']= Y_train
#h5f['Z_train']= Z_train
#h5f['X_test']= X_test
#h5f['Y_test']= Y_test
#h5f['Z_test']= Z_test
#h5f.close()


h5f = h5py.File(data_dir + 'Pancreas_CT_train_test.h5','r')
X_train= h5f['X_train'][:]
Y_train= h5f['Y_train'][:]
Z_train= h5f['Z_train'][:]
X_test= h5f['X_test'][:]
Y_test= h5f['Y_test'][:]
Z_test= h5f['Z_test'][:]
h5f.close()


X_train[X_train>1000]  = 1000
X_train[X_train<-1000] = -1000
X_train += 1000
X_train /= 2000
X_test[X_test>1000]  = 1000
X_test[X_test<-1000] = -1000
X_test += 1000
X_test /= 2000



n_train,_,_,_,_= X_train.shape
n_test,_,_,_,_=  X_test.shape

sx= 128
sy= 128
sz= 72

X_valid= X_train[:10,:,:,:,:].copy()
Y_valid= Y_train[:10,:,:,:,:].copy()

n_train,_,_,_,_= X_train.shape
n_valid,_,_,_,_= X_valid.shape
n_test,_,_,_,_=  X_test.shape




def slice_and_scale(tensor, image_idx=0, slice_idx= 0, channel_idx=0):
    t= tensor[image_idx, slice_idx, :, :, channel_idx]
    t= ( t - tf.reduce_min(t) ) / tf.reduce_max(t) * 255
    t= tf.reshape( t , tf.stack((1, sx, sy, 1)))
    return t



def davood_net_ROI(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.001):
    
    
    feat_fine= [None]*(depth-1)
    
    
    if write_summaries:
        tf.summary.image('input image', slice_and_scale(X))
    
        
    # Convolution Layers
    
    for level in range(depth):
        
        
        # Option 2
        
        ks= ks_0
        
        if level==0:
            
            strd= 1
            
            n_l= n_channel*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_init'
            name_b= 'b_'+ str(level) + '_init'
            name_c= 'Conv_'+ str(level) + '_init'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_channel,n_feat_0], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            #inp= tf.nn.dropout(inp, p_keep_conv)
        
        else:
        
            strd= 2
            
            n_l= n_channel*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_init'
            name_b= 'b_'+ str(level) + '_init'
            name_c= 'Conv_'+ str(level) + '_init'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_channel,n_feat_0], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(X, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            for i in range(1,level):
                
                n_l= n_feat_0*ks**3
                s_dev= np.sqrt(2.0/n_l)
                name_w= 'W_'+ str(level) + '_' + str(i) + '_init'
                name_b= 'b_'+ str(level) + '_' + str(i) + '_init'
                name_c= 'Conv_'+ str(level) + '_' + str(i) + '_init'
                W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat_0,n_feat_0], stddev=s_dev), name=name_w)
                b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat_0]), name=name_b)
                inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
                #inp= tf.nn.dropout(inp, p_keep_conv)
            
            for level_reg in range(0, level):
                
                inp_0= feat_fine[level_reg]
                
                level_diff= level- level_reg
                
                n_feat= n_feat_0 * 2**level_reg
                n_l= n_feat*ks**3
                s_dev= np.sqrt(2.0/n_l)
                
                for j in range(level_diff):
                    
                    name_w= 'W_'+ str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_b= 'b_'+ str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    name_c= 'Conv_'+ str(level) + '_' + str(level_reg) + '_' + str(j) + '_reg'
                    W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
                    b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
                    inp_0 = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp_0, W_1, strides=[1, strd,strd,strd, 1], padding='SAME'), b_1))
                    #inp_0 = tf.nn.dropout(inp_0, p_keep_conv)
                    
                inp= tf.concat([inp, inp_0], 4)
            
                
        ks= ks_0
        
        n_feat= n_feat_0 * 2**level
        
        
        if level>-1:
            
            inp_0= inp   ###
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_2_down'
            name_b= 'b_'+ str(level) + '_2_down'
            name_c= 'Conv_'+ str(level) + '_2_down'
            W_2= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
            b_2= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_3_down'
            name_b= 'b_'+ str(level) + '_3_down'
            name_c= 'Conv_'+ str(level) + '_3_down'
            W_3= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
            b_3= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
        
            inp= inp + inp_0  ###
        
        
        if level>-1:
            
            inp_1= inp   ###
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_4_down'
            name_b= 'b_'+ str(level) + '_4_down'
            name_c= 'Conv_'+ str(level) + '_4_down'
            W_2= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
            b_2= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_5_down'
            name_b= 'b_'+ str(level) + '_5_down'
            name_c= 'Conv_'+ str(level) + '_5_down'
            W_3= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
            b_3= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
        
            inp= inp + inp_1  + inp_0 ###
        
        
        if level<depth-1:
            feat_fine[level]= inp
        
    
    
    # DeConvolution Layers
    
    for level in range(depth-2,-1,-1):
        
        ks= ks_0
        
        n_l= n_feat*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(level) + '_up'
        name_b= 'b_'+ str(level) + '_up'
        name_c= 'Conv_'+ str(level) + '_up'
        W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat//2,n_feat], stddev=s_dev), name=name_w)
        b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_feat//2]), name=name_b)
        in_shape = tf.shape(inp)
        if level==3:
            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2,      9      , in_shape[4]//2])
        else:
            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, in_shape[4]//2])
        Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
        Deconv= tl.activation.leaky_relu(tf.add(Deconv, b_deconv))
        #Deconv= tf.nn.dropout(Deconv, p_keep_conv)
        inp= tf.concat([feat_fine[level], Deconv], 4)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_deconv)
            tf.summary.histogram(name_b, b_deconv)
            tf.summary.image(name_c, slice_and_scale(inp))
        
        if level == depth-2:
            n_concat= n_feat
        else:
            n_concat= n_feat*3//4
            
        if level < depth-2:
            n_feat= n_feat//2
            
        n_l= n_concat*ks**3
        s_dev= np.sqrt(2.0/n_l)
        name_w= 'W_'+ str(level) + '_1_up'
        name_b= 'b_'+ str(level) + '_1_up'
        name_c= 'Conv_'+ str(level) + '_1_up'
        W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_concat,n_feat], stddev=s_dev), name=name_w)
        b_1= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
        inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1))
        #inp= tf.nn.dropout(inp, p_keep_conv)
        
        if write_summaries:
            tf.summary.histogram(name_w, W_1)
            tf.summary.histogram(name_b, b_1)
            tf.summary.image(name_c, slice_and_scale(inp))
        
        
        if level>-1:
            
            inp_0= inp   ###
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_2_up'
            name_b= 'b_'+ str(level) + '_2_up'
            name_c= 'Conv_'+ str(level) + '_2_up'
            W_2= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
            b_2= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_3_up'
            name_b= 'b_'+ str(level) + '_3_up'
            name_c= 'Conv_'+ str(level) + '_3_up'
            W_3= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
            b_3= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
            
            inp= inp + inp_0  ###
        
        
        if level>-1:
            
            inp_1= inp   ###
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_4_up'
            name_b= 'b_'+ str(level) + '_4_up'
            name_c= 'Conv_'+ str(level) + '_4_up'
            W_2= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
            b_2= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_2, strides=[1, 1, 1, 1, 1], padding='SAME'), b_2))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_2)
                tf.summary.histogram(name_b, b_2)
                tf.summary.image(name_c, slice_and_scale(inp))
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_'+ str(level) + '_5_up'
            name_b= 'b_'+ str(level) + '_5_up'
            name_c= 'Conv_'+ str(level) + '_5_up'
            W_3= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_feat], stddev=s_dev), name=name_w)
            b_3= tf.Variable(tf.constant(bias_init, shape=[n_feat]), name=name_b)
            inp = tl.activation.leaky_relu(tf.add(tf.nn.conv3d(inp, W_3, strides=[1, 1, 1, 1, 1], padding='SAME'), b_3))
            #inp= tf.nn.dropout(inp, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_3)
                tf.summary.histogram(name_b, b_3)
                tf.summary.image(name_c, slice_and_scale(inp))
            
            inp= inp + inp_1 + inp_0  ###
        
        
        if level==0:
            
#            n_l= n_feat*ks**3
#            s_dev= np.sqrt(2.0/n_l)
#            name_w= 'W_up'
#            name_b= 'b_up'
#            name_c= 'Conv_up'
#            W_deconv= tf.Variable(tf.truncated_normal([ks,ks,ks,n_class,n_feat], stddev=s_dev), name=name_w)
#            b_deconv= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
#            in_shape = tf.shape(inp)
#            out_shape = tf.stack([in_shape[0], in_shape[1]*2, in_shape[2]*2, in_shape[3]*2, n_class])
#            Deconv= tf.nn.conv3d_transpose(inp, W_deconv, out_shape, strides=[1,2,2,2,1], padding='SAME')
#            output= tf.add(Deconv, b_deconv)
            
            n_l= n_feat*ks**3
            s_dev= np.sqrt(2.0/n_l)
            name_w= 'W_out'
            name_b= 'b_out'
            name_c= 'Conv_out'
            W_1= tf.Variable(tf.truncated_normal([ks,ks,ks,n_feat,n_class], stddev=s_dev), name=name_w)
            b_1= tf.Variable(tf.constant(bias_init, shape=[n_class]), name=name_b)
            output = tf.add(tf.nn.conv3d(inp, W_1, strides=[1, 1, 1, 1, 1], padding='SAME'), b_1)
            #output= tf.nn.dropout(output, p_keep_conv)
            
            if write_summaries:
                tf.summary.histogram(name_w, W_1)
                tf.summary.histogram(name_b, b_1)
                tf.summary.image(name_c, slice_and_scale(inp))
        
    
    return output




ks_0= 3
n_feat_0= 28 #36
depth= 5
n_channel= 1
n_class= 2

batch_size = 1
n_epochs = 1000
#L_Rate = 9e-6
#cost_old= 10**10
cost_fcn= 'Dice'
dice_channel= 1

restore_model= False
save_model=    True
write_summaries= False


#summaries_path=     '/media/davood/dkres/Pancreas_CT/save/'
#restore_model_path= '/media/davood/dkres/Pancreas_CT/model_checkpoint/promise_2963.ckpt'
#save_model_path=    '/media/davood/dkres/Pancreas_CT/model_checkpoint/'



X = tf.placeholder("float32", [None, sx, sy, sz, n_channel])
Y = tf.placeholder("float32", [None, sx, sy, sz, n_class])
learning_rate = tf.placeholder("float")
p_keep_conv   = tf.placeholder("float")
#p_keep_hidden = tf.placeholder("float")



softmax_linear= davood_net_ROI(X, ks_0, depth, n_feat_0, n_channel, n_class, p_keep_conv, bias_init=0.0)

predicter= tf.nn.softmax(softmax_linear)


#Dice_weight= np.zeros( (1,sx,sy,sz) )
#for i in range(sz):
#    Dice_weight[0,:,:,i]= 1 + 2 * ( abs(i-sz/2+1)/sz )**0.5
#Dice_weight[Dice_weight>2.0] = 2.0

if cost_fcn== 'Dice':
    L_Rate = 0.1e-4
    label_dice= Y[:,:,:,:,dice_channel] #* Dice_weight
    pred_dice=  predicter[:,:,:,:,dice_channel] #* Dice_weight
    cost = 1 - tl.cost.dice_coe(pred_dice, label_dice)
#    cost = 1.0 - tl.cost.dice_coe(pred_dice, label_dice, loss_type='sorensen')
#    label_dice= Y
#    pred_dice=  predicter
#    cost = 1 - tl.cost.dice_coe(pred_dice, label_dice)
elif cost_fcn== 'Our_Dice':
    L_Rate = 12e-6
    smooth=1e-5
    label_dice= Y[:,:,:,:,dice_channel]# * Dice_weight
    pred_dice=  predicter[:,:,:,:,dice_channel]# * Dice_weight
    inse = tf.reduce_sum(label_dice * pred_dice)
#    l = tf.reduce_sum(label_dice * label_dice)
#    r = tf.reduce_sum(pred_dice * pred_dice)
    l = tf.reduce_sum(label_dice*label_dice)
    r = tf.reduce_sum(pred_dice*pred_dice)
    dice = (2.0 * inse + smooth) / (l + r + smooth)
    ##
    cost = 1 - tf.reduce_mean(dice)
elif cost_fcn== 'Tversky':
    L_Rate = 1e-4
    smooth=1e-5
    label_dice= Y[:,:,:,:,dice_channel]# * Dice_weight
    pred_dice=  predicter[:,:,:,:,dice_channel]# * Dice_weight
    inse = tf.reduce_sum(label_dice * pred_dice)
    l = tf.reduce_sum(label_dice*(1-pred_dice))
    r = tf.reduce_sum(pred_dice* (1-label_dice))
    dice = (inse + smooth) / (inse + 0.35 * l + 0.65 * r + smooth)
    ##
    cost = 1 - tf.reduce_mean(dice)
elif cost_fcn== 'Davood':
    L_Rate = 0.5e-4
    smooth=1e-5
    ##
    label_dice= Y[:,:,:,:,dice_channel]
    pred_dice=  predicter[:,:,:,:,dice_channel]
    inse = tf.reduce_sum(label_dice * pred_dice)
    l = tf.reduce_sum(label_dice*(1-pred_dice))
    r = tf.reduce_sum(pred_dice* (1-label_dice))
    dice_whole = (inse + smooth) / (inse + 0.50 * l + 0.50 * r + smooth)
    ##
    label_dice_base= Y[:,:,:,:24,dice_channel]
    pred_dice_base=  predicter[:,:,:,:24,dice_channel]
    inse_base = tf.reduce_sum(label_dice_base * pred_dice_base)
    l_base = tf.reduce_sum(label_dice_base*(1-pred_dice_base))
    r_base = tf.reduce_sum(pred_dice_base*(1-label_dice_base))
    dice_base = (inse_base + smooth) / (inse_base + 0.50 * l_base + 0.50 * r_base + smooth)
    ##
    label_dice_apex= Y[:,:,:,-24:,dice_channel]
    pred_dice_apex=  predicter[:,:,:,-24:,dice_channel]
    inse_apex = tf.reduce_sum(label_dice_apex * pred_dice_apex)
    l_apex = tf.reduce_sum(label_dice_apex*(1-pred_dice_apex))
    r_apex = tf.reduce_sum(pred_dice_apex*(1-label_dice_apex))
    dice_apex = (inse_apex + smooth) / (inse_apex + 0.50 * l_apex + 0.50 * r_apex + smooth)
    ##
    cost = 3.0 - 3.0* tf.reduce_mean(dice_whole) #- 0.3 * tf.reduce_mean(dice_base) - 0.3 * tf.reduce_mean(dice_apex)
elif cost_fcn== 'Davood2':
    L_Rate = 1.2e-4
    smooth=1e-5
    ##
    label_dice= Y[:,:,:,:,dice_channel]# * Dice_weight
    pred_dice=  predicter[:,:,:,:,dice_channel]# * Dice_weight
    dice_whole = tl.cost.dice_coe(pred_dice, label_dice)
    ##
    label_dice_base= Y[:,:,:,6:24,dice_channel]# * Dice_weight
    pred_dice_base=  predicter[:,:,:,6:24,dice_channel]# * Dice_weight
    dice_base = tl.cost.dice_coe(pred_dice_base, label_dice_base)
    ##
    label_dice_apex= Y[:,:,:,-24:-6,dice_channel]# * Dice_weight
    pred_dice_apex=  predicter[:,:,:,-24:-6,dice_channel]# * Dice_weight
    dice_apex = tl.cost.dice_coe(pred_dice_apex, label_dice_apex)
    ##
    cost = 3.0 - tf.reduce_mean(dice_whole) - 1.0 * tf.reduce_mean(dice_base) - 1.0 * tf.reduce_mean(dice_apex)
elif cost_fcn== 'Davood3':
    L_Rate = 0.5e-6
    smooth=1e-5
    ##
    label_dice= Y[:,:,:,:,:]
    pred_dice=  predicter[:,:,:,:,:]
    inse = tf.reduce_sum(label_dice * pred_dice)
    l = tf.reduce_sum(label_dice*(1-pred_dice))
    r = tf.reduce_sum(pred_dice* (1-label_dice))
    dice_whole = (inse + smooth) / (inse + 0.50 * l + 0.50 * r + smooth)
    ##
    label_dice_base= Y[:,:,:,:24,:]
    pred_dice_base=  predicter[:,:,:,:24,:]
    inse_base = tf.reduce_sum(label_dice_base * pred_dice_base)
    l_base = tf.reduce_sum(label_dice_base*(1-pred_dice_base))
    r_base = tf.reduce_sum(pred_dice_base*(1-label_dice_base))
    dice_base = (inse_base + smooth) / (inse_base + 0.50 * l_base + 0.50 * r_base + smooth)
    ##
    label_dice_apex= Y[:,:,:,-24:,:]
    pred_dice_apex=  predicter[:,:,:,-24:,:]
    inse_apex = tf.reduce_sum(label_dice_apex * pred_dice_apex)
    l_apex = tf.reduce_sum(label_dice_apex*(1-pred_dice_apex))
    r_apex = tf.reduce_sum(pred_dice_apex*(1-label_dice_apex))
    dice_apex = (inse_apex + smooth) / (inse_apex + 0.50 * l_apex + 0.50 * r_apex + smooth)
    ##
    cost = 3.0 - 3.0* tf.reduce_mean(dice_whole) #- 0.3 * tf.reduce_mean(dice_base) - 0.3 * tf.reduce_mean(dice_apex)
elif cost_fcn== 'Youden':
    L_Rate = 1e-6
    smooth=1e-5
    label_dice= Y[:,:,:,:,dice_channel] #* Dice_weight
    pred_dice=  predicter[:,:,:,:,dice_channel] #* Dice_weight
    sensitivity= tf.reduce_sum( (label_dice) * (pred_dice) )  /  ( tf.reduce_sum( pred_dice ) + smooth )
    specificity= 1.0 - tf.reduce_sum( (1-label_dice) * (pred_dice) ) / ( tf.reduce_sum( pred_dice ) + smooth )
    #sensitivity= tf.reduce_sum(tf.cast((label_dice==1)*(pred_dice>0.5),tf.float32)) / (tf.reduce_sum(tf.cast(label_dice==1,tf.float32))+smooth )
    #specificity= 1.0 - tf.reduce_sum(tf.cast((label_dice==0)*(pred_dice>0.5),tf.float32))/(tf.reduce_sum(tf.cast(pred_dice>0.5,tf.float32))+smooth )
    cost= 2 - sensitivity - specificity
else:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=softmax_linear, labels=Y) )

accuracy= tf.reduce_mean( tf.cast( tf.argmax(softmax_linear, axis=-1) == tf.argmax(Y, axis=-1), tf.float32 ) )

if write_summaries:
    tf.summary.scalar("cost", cost)
    tf.summary.scalar("accuracy", accuracy)



saver = tf.train.Saver(max_to_keep=500)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#optimizer = tf.train.MomentumOptimizer(learning_rate,0.95).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


#if restore_model:
#    saver.restore(sess, restore_model_path)


#writer = tf.summary.FileWriter(summaries_path, sess.graph)
#merged = tf.summary.merge_all()


inclusion_threshold= 0.0
test_interval= 400
i_global= -1
best_test= 10**10




tr_eval_indices= np.arange(n_train)
#tr_eval_indices= tr_eval_indices[::n_train//n_test]
vl_eval_indices= np.arange(n_valid)
te_eval_indices= np.arange(n_test)

thumbs_indices_tr=  np.array([5, 10, 15, 20, 25, 30])
thumbs_indices_te=  np.array([5, 10, 15, 20, 25, 30])
thumb_slices= np.array([24, 36, 48])


Cost_Train= np.zeros(n_epochs*n_train//test_interval)
Dice_Train= np.zeros(n_epochs*n_train//test_interval)
Spec_Train= np.zeros(n_epochs*n_train//test_interval)
Sens_Train= np.zeros(n_epochs*n_train//test_interval)
Cost_Valid= np.zeros(n_epochs*n_train//test_interval)
Dice_Valid= np.zeros(n_epochs*n_train//test_interval)
Spec_Valid= np.zeros(n_epochs*n_train//test_interval)
Sens_Valid= np.zeros(n_epochs*n_train//test_interval)
Cost_Test=  np.zeros(n_epochs*n_train//test_interval)
Dice_Test=  np.zeros(n_epochs*n_train//test_interval)
Spec_Test=  np.zeros(n_epochs*n_train//test_interval)
Sens_Test=  np.zeros(n_epochs*n_train//test_interval)
i_eval= -1


#APPLY_DEFORMATION= True
#EPOCH_BEGIN_DEFORMATION= 10
#alpha= 2.0
#APPLY_SHIFT= True
#EPOCH_BEGIN_SHIFT= 2
#shift_x= 12
#shift_y= 12
#shift_z= 6
#ADD_NOISE= True
#EPOCH_BEGIN_NOISE= 2
#noise_sigma= 0.20
APPLY_DEFORMATION= True
EPOCH_BEGIN_DEFORMATION= 50
alpha= 2.0
APPLY_SHIFT= True
EPOCH_BEGIN_SHIFT= 20
shift_x= 16
shift_y= 16
shift_z= 8
ADD_NOISE= True
EPOCH_BEGIN_NOISE= 2
noise_sigma= 0.03



for epoch_i in range(n_epochs):
    
    for i in range(n_train // batch_size):
        
        batch_x = X_train[ i*batch_size:(i+1)*batch_size , : , : , : , :].copy()
        batch_y = Y_train[ i*batch_size:(i+1)*batch_size , : , : , : , :].copy()
        
        ###  BEGIN - Data Augmentation
        
        if APPLY_DEFORMATION and epoch_i>EPOCH_BEGIN_DEFORMATION:
            
            x= batch_x[0,:,:,:,0].copy()
            y= batch_y[0,:,:,:,1].copy()
            
            x= np.transpose(x, [2,1,0])
            x= sitk.GetImageFromArray(x)
            y= np.transpose(y, [2,1,0])
            y= sitk.GetImageFromArray(y)
            
            grid_physical_spacing = [25.0, 25.0, 25.0]
            image_physical_size = [size*spacing for size,spacing in zip(x.GetSize(), x.GetSpacing())]
            mesh_size = [int(image_size/grid_spacing + 0.5) \
                         for image_size,grid_spacing in zip(image_physical_size,grid_physical_spacing)]
            
            tx = sitk.BSplineTransformInitializer(x, mesh_size)
            
            direction_size= (mesh_size[0]+3) * (mesh_size[1]+3) * (mesh_size[2]+3) * 3
            
            direction= alpha * np.random.randn(direction_size)
            tx.SetParameters(direction)
            
            xx = sitk.Resample(x, x, tx, sitk.sitkBSpline, 0.0, x.GetPixelIDValue())
            yy = sitk.Resample(y, y, tx, sitk.sitkNearestNeighbor, 0.0, y.GetPixelIDValue())
            #yy = sitk.Resample(y, y, tx, sitk.sitkLinear, 0.0, y.GetPixelIDValue())
            
            x= sitk.GetArrayFromImage(xx)
            x= np.transpose(x, [2,1,0])
            
            y= sitk.GetArrayFromImage(yy)
            y= np.transpose(y, [2,1,0])
            
            batch_x[0,:,:,:,0]= x
            #batch_y[0,:,:,:,0]= y
            #batch_y[0,:,:,:,1]= 1-y
            batch_y[0,:,:,:,0]= np.float32(y<=0.5)
            batch_y[0,:,:,:,1]= np.float32(y>0.5)
            
            x= y= xx= yy= 0
        
        if APPLY_SHIFT and epoch_i>EPOCH_BEGIN_SHIFT:
            
            x= batch_x[0,:,:,:,0].copy()
            y= batch_y[0,:,:,:,1].copy()
            
            xx= np.zeros( (sx+shift_x,sy+shift_y,sz+shift_z) )
            xx[shift_x//2:shift_x//2+sx,shift_y//2:shift_y//2+sy,shift_z//2:shift_z//2+sz]= x.copy()
            yy= np.zeros( (sx+shift_x,sy+shift_y,sz+shift_z) )
            yy[shift_x//2:shift_x//2+sx,shift_y//2:shift_y//2+sy,shift_z//2:shift_z//2+sz]= y.copy()
            
            shift_xx= np.random.randint(shift_x)
            shift_yy= np.random.randint(shift_y)
            shift_zz= np.random.randint(shift_z)
            
            batch_x[0,:,:,:,0]=   xx[shift_xx:shift_xx+sx, shift_yy:shift_yy+sy, shift_zz:shift_zz+sz].copy()
            batch_y[0,:,:,:,0]= 1-yy[shift_xx:shift_xx+sx, shift_yy:shift_yy+sy, shift_zz:shift_zz+sz].copy()
            batch_y[0,:,:,:,1]=   yy[shift_xx:shift_xx+sx, shift_yy:shift_yy+sy, shift_zz:shift_zz+sz].copy()
            
            x= y= xx= yy= 0
            
        if ADD_NOISE and epoch_i>EPOCH_BEGIN_NOISE:
            
            additive_noise= np.random.randn( batch_size, sx, sy, sz, n_channel ) * noise_sigma
            batch_x+= additive_noise
        
        ###  END - Data Augmentation
        
        if np.mean(batch_y[:,:,:,:,0]==1)>inclusion_threshold:
#            print (i, np.mean(batch_y[:,:,:,:,0]==1) )
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y, learning_rate: L_Rate, p_keep_conv: 1.0})
            batch_x= batch_y= 0
        
        i_global+= 1
        
        if i_global%test_interval==0:
            
            print(epoch_i, i, i_global)
            
            train_cost= np.zeros(len(tr_eval_indices))
            train_dice= np.zeros(len(tr_eval_indices))
            tr_sen=     np.zeros(len(tr_eval_indices))
            tr_spc=     np.zeros(len(tr_eval_indices))
            '''valid_cost= np.zeros(len(vl_eval_indices))
            valid_dice= np.zeros(len(vl_eval_indices))
            vl_sen=     np.zeros(len(vl_eval_indices))
            vl_spc=     np.zeros(len(vl_eval_indices))'''
            test_cost=  np.zeros(len(te_eval_indices))
            test_dice=  np.zeros(len(te_eval_indices))
            te_sen=     np.zeros(len(te_eval_indices))
            te_spc=     np.zeros(len(te_eval_indices))
            
            for i_tr in range(len(tr_eval_indices)):
                
                train_ind= tr_eval_indices[i_tr]
                batch_x = X_train[ train_ind*batch_size:(train_ind+1)*batch_size , : , : , : , :].copy()
                batch_y = Y_train[ train_ind*batch_size:(train_ind+1)*batch_size , : , : , : , :].copy()
                train_cost[i_tr] = sess.run(cost, feed_dict={X: batch_x, Y: batch_y, p_keep_conv: 1.0})
                
                y_tr=     Y_train[ train_ind*batch_size:(train_ind+1)*batch_size , : , : , : , :].copy()
                batch_x = X_train[ train_ind*batch_size:(train_ind+1)*batch_size , : , : , : , :].copy()
                y_tr_pr = sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: 1.0})
                
                tr_sen[i_tr]= 100.0 * np.sum( (y_tr[:,:,:,:,1]==1) * (y_tr_pr[:,:,:,:,1]>0.5) )  / np.sum( y_tr[:,:,:,:,1]==1 )
                tr_spc[i_tr]= 100.0 - 100.0 * np.sum( (y_tr[:,:,:,:,1]==0) * (y_tr_pr[:,:,:,:,1]>0.5) ) / np.sum( y_tr_pr[:,:,:,:,1]>0.5 )
                dice_num= 2 * np.sum( ( y_tr[:,:,:,:,1]==1 )    *    ( y_tr_pr[:,:,:,:,1]>0.5 )  )
                dice_den=     np.sum(   y_tr[:,:,:,:,1]==1 ) + np.sum( y_tr_pr[:,:,:,:,1]>0.5 )
                train_dice[i_tr] = dice_num/dice_den
            
            '''for i_vl in range(len(vl_eval_indices)):
                
                valid_ind= vl_eval_indices[i_vl]
                batch_x = X_valid[ valid_ind*batch_size:(valid_ind+1)*batch_size , : , : , : , :].copy()
                batch_y = Y_valid[ valid_ind*batch_size:(valid_ind+1)*batch_size , : , : , : , :].copy()
                valid_cost[i_vl] = sess.run(cost, feed_dict={X: batch_x, Y: batch_y, p_keep_conv: 1.0})
                
                y_vl=     Y_valid[ valid_ind*batch_size:(valid_ind+1)*batch_size , : , : , : , :].copy()
                batch_x = X_valid[ valid_ind*batch_size:(valid_ind+1)*batch_size , : , : , : , :].copy()
                y_vl_pr = sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: 1.0})
                
                vl_sen[i_vl]= 100.0 * np.sum( (y_vl[:,:,:,:,1]==1) * (y_vl_pr[:,:,:,:,1]>0.5) )  / np.sum( y_vl[:,:,:,:,1]==1 )
                vl_spc[i_vl]= 100.0 - 100.0 * np.sum( (y_vl[:,:,:,:,1]==0) * (y_vl_pr[:,:,:,:,1]>0.5) ) / np.sum( y_vl_pr[:,:,:,:,1]>0.5 )
                dice_num= 2 * np.sum( ( y_vl[:,:,:,:,1]==1 )    *    ( y_vl_pr[:,:,:,:,1]>0.5 )  )
                dice_den=     np.sum(   y_vl[:,:,:,:,1]==1 ) + np.sum( y_vl_pr[:,:,:,:,1]>0.5 )
                valid_dice[i_vl] = dice_num/dice_den
            '''    
            for i_te in range(len(te_eval_indices)):
                
                test_ind= te_eval_indices[i_te]
                batch_x = X_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :].copy()
                batch_y = Y_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :].copy()
                test_cost[i_te] = sess.run(cost, feed_dict={X: batch_x, Y: batch_y, p_keep_conv: 1.0})
                
                y_te=     Y_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :].copy()
                batch_x = X_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :].copy()
                
                y_te_pr = sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: 1.0})
                te_sen[i_te]= 100.0 * np.sum( (y_te[:,:,:,:,1]==1) * (y_te_pr[:,:,:,:,1]>0.5) )  / np.sum( y_te[:,:,:,:,1]==1 )
                te_spc[i_te]= 100.0 - 100.0 * np.sum( (y_te[:,:,:,:,1]==0) * (y_te_pr[:,:,:,:,1]>0.5) ) / np.sum( y_te_pr[:,:,:,:,1]>0.5 )
                dice_num= 2 * np.sum( ( y_te[:,:,:,:,1]==1 )    *    ( y_te_pr[:,:,:,:,1]>0.5 ) )
                dice_den=     np.sum(   y_te[:,:,:,:,1]==1 ) + np.sum( y_te_pr[:,:,:,:,1]>0.5 )
                test_dice[i_te] = dice_num/dice_den
                
            print ('train cost and dice   %.3f' % train_cost.mean(), ', %.3f' % train_dice.mean())
            #print ('valid cost and dice   %.3f' % valid_cost.mean(), ', %.3f' % valid_dice.mean())
            print ('test  cost and dice   %.3f' % test_cost.mean(), ', %.3f' % test_dice.mean())
            print ('train sens and spec   %.2f' % tr_sen.mean(), ', %.2f' % tr_spc.mean())
            #print ('valid sens and spec   %.2f' % vl_sen.mean(), ', %.2f' % vl_spc.mean())
            print ('test  sens and spec   %.2f' % te_sen.mean(), ', %.2f' % te_spc.mean())
            
            if True: #test_cost.mean()<best_test:
                best_test= test_cost.mean()
                temp_path= data_dir + 'promise_'+ str(int(round(10000.0*train_cost.mean()))) \
                + '_' + str(int(round(100.0*train_dice.mean()))) \
                + '_' + str(int(round(100.0*test_dice.mean()))) + '.ckpt'
                saver.save(sess, temp_path)
            
            if epoch_i==0 and i==0:
                cost_old= train_cost.mean()
            else:
                if train_cost.mean()>0.99 * cost_old:
                    L_Rate= L_Rate * 0.995
                cost_old= train_cost.mean()
                
            print ('learning rate:  ', L_Rate)
            
            i_eval+= 1
            Cost_Train[i_eval]= train_cost.mean()
            Dice_Train[i_eval]= train_dice.mean()
            Spec_Train[i_eval]= tr_spc.mean()
            Sens_Train[i_eval]= tr_sen.mean()
            Cost_Test[i_eval]= test_cost.mean()
            Dice_Test[i_eval]= test_dice.mean()
            Spec_Test[i_eval]= te_spc.mean()
            Sens_Test[i_eval]= te_sen.mean()
            '''
            if i_eval==0:
                for file in os.listdir(thmbs_dir):
                    file_path = os.path.join(thmbs_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)
            
            for i_th in thumbs_indices_tr:
                
                x_th= X_train[i_th*batch_size:(i_th+1)*batch_size , : , : , : , :].copy()
                y_th= Y_train[i_th*batch_size:(i_th+1)*batch_size , : , : , : , :].copy()
                
                y_th_pr= sess.run(predicter, feed_dict={X: x_th, p_keep_conv: 1.0})
                
                if i_global==0:
                    for slc in thumb_slices:
                        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
                        ax.imshow(x_th[0,:,:,slc,0], cmap='gray')
                        fig.savefig(thmbs_dir + 'Y_train_' + str(i_th) + '_' + str(slc) + '_fig.png') 
                        plt.close(fig)
                        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
                        ax.imshow(y_th[0,:,:,slc,1], cmap='gray')
                        fig.savefig(thmbs_dir + 'Y_train_' + str(i_th) + '_' + str(slc) + '_gold.png') 
                        plt.close(fig)
                    file_name= 'Tr_' + str(i_th) + '_image.mhd'
                    x= x_th[0,:,:,:,0]
                    x= np.transpose(x, [2,1,0])
                    x= sitk.GetImageFromArray(x)
                    sitk.WriteImage(x, mhd_dir + file_name )
                    file_name= 'Tr_' + str(i_th) + '_seg_0.mhd'
                    x= y_th[0,:,:,:,1]
                    x= np.transpose(x, [2,1,0])
                    x= sitk.GetImageFromArray(x)
                    sitk.WriteImage(x, mhd_dir + file_name )
                else:
                    for slc in thumb_slices:
                        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
                        ax.imshow(y_th_pr[0,:,:,slc,1], cmap='gray')
                        fig.savefig(thmbs_dir + 'Y_train_' + str(i_th) + '_' + str(slc) + '_pred' + '_' + str(i_eval) + '.png')
                        plt.close(fig)
                    file_name= 'Tr_' + str(i_th) + '_seg_'+str(i_eval)+'.mhd'
                    x= y_th_pr[0,:,:,:,1]
                    x= np.transpose(x, [2,1,0])
                    x= sitk.GetImageFromArray(x)
                    sitk.WriteImage(x, mhd_dir + file_name )
                    file_name= 'Tr_' + str(i_th) + '_seg_'+str(i_eval)+'_hard.mhd'
                    x= np.float32(y_th_pr[0,:,:,:,1]>0.50)
                    x= np.transpose(x, [2,1,0])
                    x= sitk.GetImageFromArray(x)
                    sitk.WriteImage(x, mhd_dir + file_name )
                    
            for i_th in thumbs_indices_te:
                
                x_th= X_test[i_th*batch_size:(i_th+1)*batch_size,:,:,:,:].copy()
                y_th= Y_test[i_th*batch_size:(i_th+1)*batch_size,:,:,:,:].copy()
                
                y_th_pr= sess.run(predicter, feed_dict={X: x_th, p_keep_conv: 1.0})
                
                if i_global==0:
                    for slc in thumb_slices:
                        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
                        ax.imshow(x_th[0,:,:,slc,0], cmap='gray')
                        fig.savefig(thmbs_dir + 'Y_test_' + str(i_th) + '_' + str(slc) + '_fig.png') 
                        plt.close(fig)
                        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
                        ax.imshow(y_th[0,:,:,slc,1], cmap='gray')
                        fig.savefig(thmbs_dir + 'Y_test_' + str(i_th) + '_' + str(slc) + '_gold.png') 
                        plt.close(fig)
                    file_name= 'Te_' + str(i_th) + '_image.mhd'
                    x= x_th[0,:,:,:,0]
                    x= np.transpose(x, [2,1,0])
                    x= sitk.GetImageFromArray(x)
                    sitk.WriteImage(x, mhd_dir + file_name )
                    file_name= 'Te_' + str(i_th) + '_seg_0.mhd'
                    x= y_th[0,:,:,:,1]
                    x= np.transpose(x, [2,1,0])
                    x= sitk.GetImageFromArray(x)
                    sitk.WriteImage(x, mhd_dir + file_name )
                else:
                    for slc in thumb_slices:
                        fig, ax = plt.subplots( nrows=1, ncols=1 ) 
                        ax.imshow(y_th_pr[0,:,:,slc,1], cmap='gray')
                        fig.savefig(thmbs_dir + 'Y_test_' + str(i_th) + '_' + str(slc) + '_pred' + '_' + str(i_eval) + '.png')  
                        plt.close(fig)
                    file_name= 'Te_' + str(i_th) + '_seg_'+str(i_eval)+'.mhd'
                    x= y_th_pr[0,:,:,:,1]
                    x= np.transpose(x, [2,1,0])
                    x= sitk.GetImageFromArray(x)
                    sitk.WriteImage(x, mhd_dir + file_name )
                    file_name= 'Te_' + str(i_th) + '_seg_'+str(i_eval)+'_hard.mhd'
                    x= np.float32(y_th_pr[0,:,:,:,1]>0.50)
                    x= np.transpose(x, [2,1,0])
                    x= sitk.GetImageFromArray(x)
                    sitk.WriteImage(x, mhd_dir + file_name )
                    '''

sio.savemat(data_dir + 'res_01.mat', {'Dice_Train':Dice_Train, 'Dice_Test':Dice_Test})


#######################################################################################

#
#X_train= np.concatenate( (X_train, X_train, X_train, X_test), axis=0 )
#Y_train= np.concatenate( (Y_train, Y_train, Y_train, Y_test), axis=0 )
#n_train,_,_,_,_= X_train.shape
#
#######################################################################################
#plt.figure(), plt.plot(Cost_Train[:i_eval],'b'), plt.plot(Cost_Test[:i_eval],'.-b')
#plt.figure(), plt.plot(100*Dice_Train[:i_eval],'r'), plt.plot(100*Dice_Test[:i_eval],'.-r')
#plt.plot(Sens_Train[:i_eval],'k'), plt.plot(Sens_Test[:i_eval],'.-k')
#plt.plot(Spec_Train[:i_eval],'b'), plt.plot(Spec_Test[:i_eval],'.-b')
#######################################################################################






#######################################################################################
#######################################################################################
#######################################################################################






'''
#restore_model_path= '/media/davood/New Volume/PROMISE12/submission/Submission4/checkpoints/promise_964.ckpt'
#restore_model_path= 'G:\\PROMISE12\\submission\\Submission2\\checkpoints\\promise_1078.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/01/promise_3013.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/02/promise_3016.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/03/promise_2974.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/04/promise_2995.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/05/promise_2941.ckpt'

save_intermed_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/05/intermed/'

saver.restore(sess, restore_model_path)

results_summary= np.zeros( (n_test, 12 ) )



# Option 1 - no shift

for test_ind in range(n_test):
    
    batch_x = X_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :]   
    y_te=     Y_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :]
    
    y_te_pr = sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: 1.0})
    
    te_sen= 100.0 * np.sum( (y_te[:,:,:,:,1]==1) * (y_te_pr[:,:,:,:,1]>0.5) )  / np.sum( y_te[:,:,:,:,1]==1 )
    te_spc= 100.0 - 100.0 * np.sum( (y_te[:,:,:,:,1]==0) * (y_te_pr[:,:,:,:,1]>0.5) ) / np.sum( y_te_pr[:,:,:,:,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,:,1]==1 )    *    ( y_te_pr[:,:,:,:,1]>0.5 ) )
    dice_den=     np.sum(   y_te[:,:,:,:,1]==1 ) + np.sum( y_te_pr[:,:,:,:,1]>0.5 )
    test_dice = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,:3] = [test_dice, te_sen, te_spc]
    
    te_sen_midg= 100.0 * np.sum( (y_te[:,:,:,24:-24,1]==1) * (y_te_pr[:,:,:,24:-24,1]>0.5) )  / np.sum( y_te[:,:,:,24:-24,1]==1 )
    te_spc_midg= 100.0 - 100.0 * np.sum( (y_te[:,:,:,24:-24,1]==0) * (y_te_pr[:,:,:,24:-24,1]>0.5) ) / np.sum( y_te_pr[:,:,:,24:-24,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,24:-24,1]==1 )    *    ( y_te_pr[:,:,:,24:-24,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,24:-24,1]==1 ) + np.sum( y_te_pr[:,:,:,24:-24,1]>0.5 )
    test_dice_midg = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,3:6] = [test_dice_midg, te_sen_midg, te_spc_midg]
    
    te_sen_base= 100.0 * np.sum( (y_te[:,:,:,6:24,1]==1) * (y_te_pr[:,:,:,6:24,1]>0.5) )  / np.sum( y_te[:,:,:,6:24,1]==1 )
    te_spc_base= 100.0 - 100.0 * np.sum( (y_te[:,:,:,6:24,1]==0) * (y_te_pr[:,:,:,6:24,1]>0.5) ) / np.sum( y_te_pr[:,:,:,6:24,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,6:24,1]==1 )    *    ( y_te_pr[:,:,:,6:24,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,6:24,1]==1 ) + np.sum( y_te_pr[:,:,:,6:24,1]>0.5 )
    test_dice_base = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,6:9] = [test_dice_base, te_sen_base, te_spc_base]
    
    te_sen_apex= 100.0 * np.sum( (y_te[:,:,:,-24:-6,1]==1) * (y_te_pr[:,:,:,-24:-6,1]>0.5) )  / np.sum( y_te[:,:,:,-24:-6,1]==1 )
    te_spc_apex= 100.0 - 100.0 * np.sum( (y_te[:,:,:,-24:-6,1]==0) * (y_te_pr[:,:,:,-24:-6,1]>0.5) ) / np.sum( y_te_pr[:,:,:,-24:-6,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,-24:-6,1]==1 )    *    ( y_te_pr[:,:,:,-24:-6,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,-24:-6,1]==1 ) + np.sum( y_te_pr[:,:,:,-24:-6,1]>0.5 )
    test_dice_apex = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,9:] = [test_dice_apex, te_sen_apex, te_spc_apex]
    
    #x= y_te_pr[0,:,:,:,1]
    file_name= 'Te_' + str(test_ind) + '.mhd'
    x= np.zeros(y_te_pr[0,:,:,:,1].shape)
    x[np.logical_and(y_te_pr[0,:,:,:,1]>0.5 , y_te[0,:,:,:,1]>0.5)]= 1
    x[np.logical_and(y_te_pr[0,:,:,:,1]>0.5 , y_te[0,:,:,:,1]<0.5)]= 2
    x[np.logical_and(y_te_pr[0,:,:,:,1]<0.5 , y_te[0,:,:,:,1]>0.5)]= 3
    x= np.transpose(x, [2,1,0])
    x= sitk.GetImageFromArray(x)
    sitk.WriteImage(x, save_intermed_path + 'err/' + file_name )
    
    file_name= 'Te_' + str(test_ind) + '_soft.mhd'
    x= y_te_pr[0,:,:,:,1]
    x= np.transpose(x, [2,1,0])
    x= sitk.GetImageFromArray(x)
    sitk.WriteImage(x, save_intermed_path + file_name )
    
    file_name= 'Te_' + str(test_ind) + '_hard.mhd'
    x= np.zeros(y_te_pr[0,:,:,:,1].shape)
    x[y_te_pr[0,:,:,:,1]>0.5]= 1
    x= np.transpose(x, [2,1,0])
    x= sitk.GetImageFromArray(x)
    sitk.WriteImage(x, save_intermed_path +  file_name )
    
    file_name= 'Te_' + str(test_ind) + '_image.mhd'
    x= batch_x[0,:,:,:,0]
    x= np.transpose(x, [2,1,0])
    x= sitk.GetImageFromArray(x)
    sitk.WriteImage(x, save_intermed_path + file_name )


sio.savemat(save_intermed_path + 'res_summary_05.mat', {'x_05':results_summary})




# Option 2 - do shift


restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/01/promise_3013.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/02/promise_3016.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/03/promise_2974.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/04/promise_2995.ckpt'
restore_model_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/05/promise_2941.ckpt'

save_intermed_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/01/intermedF/'

saver.restore(sess, restore_model_path)

results_summary= np.zeros( (n_test, 12 ) )


shift_x= 12
shift_y= 12
shift_z= 6

x_shifts= [0, 6, 12] #[4,6,8]#[3, 6, 9]
y_shifts= [0, 6, 12] #[4,6,8]#[3, 6, 9]
z_shifts= [0, 3, 6]#[3]#[2, 3, 4]

for test_ind in range(n_test):
    
    batch_x = X_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :]
    y_te=     Y_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :]
    
    y_te_pr= np.zeros( y_te.shape )
    y_te_pr_count= 0
    
    for x_shift in x_shifts:
        for y_shift in y_shifts:
            for z_shift in z_shifts:
                
                batch_x = X_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :].copy()
                x= batch_x[0,:,:,:,0].copy()
                
                xx= np.zeros( (sx+shift_x,sy+shift_y,sz+shift_z) )
                xx[shift_x//2:shift_x//2+sx,shift_y//2:shift_y//2+sy,shift_z//2:shift_z//2+sz]= x.copy()
                
                batch_x[0,:,:,:,0]=   xx[x_shift:x_shift+sx, y_shift:y_shift+sy, z_shift:z_shift+sz].copy()
                
                y_te_pr       += sess.run(predicter, feed_dict={X: batch_x, p_keep_conv: 1.0})
                y_te_pr_count += 1
                
    y_te_pr /= y_te_pr_count
    
    te_sen= 100.0 * np.sum( (y_te[:,:,:,:,1]==1) * (y_te_pr[:,:,:,:,1]>0.5) )  / np.sum( y_te[:,:,:,:,1]==1 )
    te_spc= 100.0 - 100.0 * np.sum( (y_te[:,:,:,:,1]==0) * (y_te_pr[:,:,:,:,1]>0.5) ) / np.sum( y_te_pr[:,:,:,:,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,:,1]==1 )    *    ( y_te_pr[:,:,:,:,1]>0.5 ) )
    dice_den=     np.sum(   y_te[:,:,:,:,1]==1 ) + np.sum( y_te_pr[:,:,:,:,1]>0.5 )
    test_dice = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,:3] = [test_dice, te_sen, te_spc]
    
    te_sen_midg= 100.0 * np.sum( (y_te[:,:,:,24:-24,1]==1) * (y_te_pr[:,:,:,24:-24,1]>0.5) )  / np.sum( y_te[:,:,:,24:-24,1]==1 )
    te_spc_midg= 100.0 - 100.0 * np.sum( (y_te[:,:,:,24:-24,1]==0) * (y_te_pr[:,:,:,24:-24,1]>0.5) ) / np.sum( y_te_pr[:,:,:,24:-24,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,24:-24,1]==1 )    *    ( y_te_pr[:,:,:,24:-24,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,24:-24,1]==1 ) + np.sum( y_te_pr[:,:,:,24:-24,1]>0.5 )
    test_dice_midg = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,3:6] = [test_dice_midg, te_sen_midg, te_spc_midg]
    
    te_sen_base= 100.0 * np.sum( (y_te[:,:,:,6:24,1]==1) * (y_te_pr[:,:,:,6:24,1]>0.5) )  / np.sum( y_te[:,:,:,6:24,1]==1 )
    te_spc_base= 100.0 - 100.0 * np.sum( (y_te[:,:,:,6:24,1]==0) * (y_te_pr[:,:,:,6:24,1]>0.5) ) / np.sum( y_te_pr[:,:,:,6:24,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,6:24,1]==1 )    *    ( y_te_pr[:,:,:,6:24,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,6:24,1]==1 ) + np.sum( y_te_pr[:,:,:,6:24,1]>0.5 )
    test_dice_base = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,6:9] = [test_dice_base, te_sen_base, te_spc_base]
    
    te_sen_apex= 100.0 * np.sum( (y_te[:,:,:,-24:-6,1]==1) * (y_te_pr[:,:,:,-24:-6,1]>0.5) )  / np.sum( y_te[:,:,:,-24:-6,1]==1 )
    te_spc_apex= 100.0 - 100.0 * np.sum( (y_te[:,:,:,-24:-6,1]==0) * (y_te_pr[:,:,:,-24:-6,1]>0.5) ) / np.sum( y_te_pr[:,:,:,-24:-6,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,-24:-6,1]==1 )    *    ( y_te_pr[:,:,:,-24:-6,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,-24:-6,1]==1 ) + np.sum( y_te_pr[:,:,:,-24:-6,1]>0.5 )
    test_dice_apex = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,9:] = [test_dice_apex, te_sen_apex, te_spc_apex]
    
    #x= y_te_pr[0,:,:,:,1]
    file_name= 'Te_' + str(test_ind) + '.mhd'
    x= np.zeros(y_te_pr[0,:,:,:,1].shape)
    x[np.logical_and(y_te_pr[0,:,:,:,1]>0.5 , y_te[0,:,:,:,1]>0.5)]= 1
    x[np.logical_and(y_te_pr[0,:,:,:,1]>0.5 , y_te[0,:,:,:,1]<0.5)]= 2
    x[np.logical_and(y_te_pr[0,:,:,:,1]<0.5 , y_te[0,:,:,:,1]>0.5)]= 3
    x= np.transpose(x, [2,1,0])
    x= sitk.GetImageFromArray(x)
    sitk.WriteImage(x, save_intermed_path + 'err/' + file_name )
    
    file_name= 'Te_' + str(test_ind) + '_soft.mhd'
    x= y_te_pr[0,:,:,:,1]
    x= np.transpose(x, [2,1,0])
    x= sitk.GetImageFromArray(x)
    sitk.WriteImage(x, save_intermed_path + file_name )
    
    file_name= 'Te_' + str(test_ind) + '_hard.mhd'
    x= np.zeros(y_te_pr[0,:,:,:,1].shape)
    x[y_te_pr[0,:,:,:,1]>0.5]= 1
    x= np.transpose(x, [2,1,0])
    x= sitk.GetImageFromArray(x)
    sitk.WriteImage(x, save_intermed_path +  file_name )
    
    file_name= 'Te_' + str(test_ind) + '_image.mhd'
    x= batch_x[0,:,:,:,0]
    x= np.transpose(x, [2,1,0])
    x= sitk.GetImageFromArray(x)
    sitk.WriteImage(x, save_intermed_path + file_name )



sio.savemat(save_intermed_path + 'res_summary_01F_wide.mat', {'x_01F':results_summary})








###############################################################################
###############################################################################
###############################################################################

####   Initial submission

save_intermed_path= '/media/davood/dkdata/PROMISE12/submission/Submission4/intermed/'

results_summary= np.zeros( (n_test, 12 ) )

for test_ind in range(n_test):
    
    y_te=     Y_test[ test_ind*batch_size:(test_ind+1)*batch_size , : , : , : , :]
    
    temp= sitk.ReadImage( save_intermed_path + 'Test_' + str(test_ind) + '.mhd' )
    temp= sitk.GetArrayFromImage(temp)
    temp= np.transpose( temp, [2,1,0])
    
    y_te_pr= np.zeros( (1, sx, sy, sz, 2) )
    y_te_pr[0,:,:,:,0] = 1-temp
    y_te_pr[0,:,:,:,1] = temp
    
    te_sen= 100.0 * np.sum( (y_te[:,:,:,:,1]==1) * (y_te_pr[:,:,:,:,1]>0.5) )  / np.sum( y_te[:,:,:,:,1]==1 )
    te_spc= 100.0 - 100.0 * np.sum( (y_te[:,:,:,:,1]==0) * (y_te_pr[:,:,:,:,1]>0.5) ) / np.sum( y_te_pr[:,:,:,:,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,:,1]==1 )    *    ( y_te_pr[:,:,:,:,1]>0.5 ) )
    dice_den=     np.sum(   y_te[:,:,:,:,1]==1 ) + np.sum( y_te_pr[:,:,:,:,1]>0.5 )
    test_dice = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,:3] = [test_dice, te_sen, te_spc]
    
    te_sen_midg= 100.0 * np.sum( (y_te[:,:,:,24:-24,1]==1) * (y_te_pr[:,:,:,24:-24,1]>0.5) )  / np.sum( y_te[:,:,:,24:-24,1]==1 )
    te_spc_midg= 100.0 - 100.0 * np.sum( (y_te[:,:,:,24:-24,1]==0) * (y_te_pr[:,:,:,24:-24,1]>0.5) ) / np.sum( y_te_pr[:,:,:,24:-24,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,24:-24,1]==1 )    *    ( y_te_pr[:,:,:,24:-24,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,24:-24,1]==1 ) + np.sum( y_te_pr[:,:,:,24:-24,1]>0.5 )
    test_dice_midg = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,3:6] = [test_dice_midg, te_sen_midg, te_spc_midg]
    
    te_sen_base= 100.0 * np.sum( (y_te[:,:,:,6:24,1]==1) * (y_te_pr[:,:,:,6:24,1]>0.5) )  / np.sum( y_te[:,:,:,6:24,1]==1 )
    te_spc_base= 100.0 - 100.0 * np.sum( (y_te[:,:,:,6:24,1]==0) * (y_te_pr[:,:,:,6:24,1]>0.5) ) / np.sum( y_te_pr[:,:,:,6:24,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,6:24,1]==1 )    *    ( y_te_pr[:,:,:,6:24,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,6:24,1]==1 ) + np.sum( y_te_pr[:,:,:,6:24,1]>0.5 )
    test_dice_base = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,6:9] = [test_dice_base, te_sen_base, te_spc_base]
    
    te_sen_apex= 100.0 * np.sum( (y_te[:,:,:,-24:-6,1]==1) * (y_te_pr[:,:,:,-24:-6,1]>0.5) )  / np.sum( y_te[:,:,:,-24:-6,1]==1 )
    te_spc_apex= 100.0 - 100.0 * np.sum( (y_te[:,:,:,-24:-6,1]==0) * (y_te_pr[:,:,:,-24:-6,1]>0.5) ) / np.sum( y_te_pr[:,:,:,-24:-6,1]>0.5 )
    dice_num= 2 * np.sum( ( y_te[:,:,:,-24:-6,1]==1 )    *    ( y_te_pr[:,:,:,-24:-6,1]>0.5 )  )
    dice_den=     np.sum(   y_te[:,:,:,-24:-6,1]==1 ) + np.sum( y_te_pr[:,:,:,-24:-6,1]>0.5 )
    test_dice_apex = 100.0 * dice_num/dice_den
    
    results_summary[test_ind,9:] = [test_dice_apex, te_sen_apex, te_spc_apex]
    
    

sio.savemat(save_intermed_path + 'res_summary_00.mat', {'x_00':results_summary})




###############################################################################
###############################################################################
###############################################################################

#res_address= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/saved/08/res.mat'
#
#sio.savemat(save_intermed_path + 'res.mat', {'test_dice':test_dice, 'te_sen':te_sen, 'te_pc':te_spc,
#                          'test_dice_base':test_dice_base, 'te_sen_base':te_sen_base, 'te_pc_base':te_spc_base,
#                          'test_dice_apex':test_dice_apex, 'te_sen_apex':te_sen_apex, 'te_pc_apex':te_spc_apex,
#                          'test_dice_midg':test_dice_midg, 'te_sen_midg':te_sen_midg, 'te_pc_midg':te_spc_midg})

###############################################################################
###############################################################################
###############################################################################



n_model= 5

Y_soft= np.zeros( (n_model, n_test, sx, sy, sz) )
Y_hard= np.zeros( (n_model, n_test, sx, sy, sz) )

for i_model in range(n_model):
    
    save_intermed_path= '/media/davood/dkdata/PROMISE12/model_checkpoint/ROI/0' + str(i_model+1) + '/intermed/'
    
    file_names = [f for f in listdir(save_intermed_path) if isfile(join(save_intermed_path, f))]
    
    for file_name in file_names:
        
        if 'soft.mhd' in file_name:
            
            y_soft= sitk.ReadImage( save_intermed_path + file_name )
            y_soft= sitk.GetArrayFromImage(y_soft)
            y_soft= np.transpose( y_soft, [2,1,0])
            
            file_name= file_name.split('Te_')[1]
            if '_' in file_name[:2]:
                file_ind= int(file_name[:1])
            else:
                file_ind= int(file_name[:2])
            
            Y_soft[i_model, file_ind, :, :, :]= y_soft.copy()
            
        elif 'hard.mhd' in file_name:
            
            y_hard= sitk.ReadImage( save_intermed_path + file_name )
            y_hard= sitk.GetArrayFromImage(y_hard)
            y_hard= np.transpose( y_hard, [2,1,0])
            
            file_name= file_name.split('Te_')[1]
            if '_' in file_name[:2]:
                file_ind= int(file_name[:1])
            else:
                file_ind= int(file_name[:2])
            
            Y_hard[i_model, file_ind, :, :, :]= y_hard.copy()

    print(i_model)




def is_boundary_point_3D(I, i, j, k):

    if I[i,j,k]==1:
    
        temp_sum= np.sum(I[i-1:i+2, j-1:j+2, k-1:k+2])
        
        if temp_sum==1 or temp_sum==27:
            return 0
        else:
            return 1
            
    else:
        
        return 0
        
Y_boundary= np.zeros( (n_test, sx, sy, sz) )

for i_test in range(n_test):
    
    y_test= Y_test[i_test,:,:,:, 1].copy()
    
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                
                if is_boundary_point_3D(y_test, i, j, k):
                    
                    Y_boundary[i_test, i, j, k]= 1
    
    print(i_test)



uncertainty_thumbs= '/media/davood/dkdata/PROMISE12/Uncertainty/thumbs/'

n_rows, n_cols= 3, 4

for i_test in range(n_test):
    
    p_hat= np.mean( Y_soft[:,i_test,:,:,:] , axis= 0 )
    p_hat_hard= (p_hat>0.5).astype(np.int)
    p_hat_boundary= np.zeros(p_hat.shape)
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                if is_boundary_point_3D(p_hat_hard, i, j, k):    
                    p_hat_boundary[i, j, k]= 1
    
    KW=    1 - p_hat**2 - (1-p_hat)**2
    KW[Y_boundary[i_test,:,:,:]==1]= 1.0
    KW[p_hat_boundary==1.0]= -0.1
    
    boundary_all= np.zeros(p_hat.shape)
    for i_model in range(n_model):
        y_hard= Y_hard[i_model,i_test,:,:,:]
        for i in range(sx):
            for j in range(sy):
                for k in range(sz):
                    if is_boundary_point_3D(y_hard, i, j, k):    
                        boundary_all[i, j, k]= i_model
    
    ######
    fig, ax = plt.subplots(figsize=(16,10), nrows= n_rows, ncols= n_cols )
    
    plt.subplot(n_rows, n_cols,1)
    plt.imshow(X_test[i_test,:,:,36,0], cmap='gray')
    plt.subplot(n_rows, n_cols,2)
    plt.imshow(Y_test[i_test,:,:,36,1], cmap='gray')
    plt.subplot(n_rows, n_cols,3)
    plt.imshow(Y_boundary[i_test,:,:,36], cmap='gray')
    
    for i_model in range(n_model):
        plt.subplot(n_rows, n_cols,4+i_model)
        plt.imshow(Y_soft[i_model,i_test,:,:,36], cmap='gray')
    
    plt.subplot(n_rows, n_cols,9)
    plt.imshow(KW[:,:,36])
    
    plt.subplot(n_rows, n_cols,10)
    plt.imshow(boundary_all[:,:,36])
    
    fig.savefig(uncertainty_thumbs + 'X_' + str(i_test) + '_axial.png')
    plt.close(fig)
    
    ######
    fig, ax = plt.subplots(figsize=(16,10), nrows= n_rows, ncols= n_cols )
    
    plt.subplot(n_rows, n_cols,1)
    plt.imshow(X_test[i_test,:,64,:,0], cmap='gray')
    plt.subplot(n_rows, n_cols,2)
    plt.imshow(Y_test[i_test,:,64,:,1], cmap='gray')
    plt.subplot(n_rows, n_cols,3)
    plt.imshow(Y_boundary[i_test,:,64,:], cmap='gray')
    
    for i_model in range(n_model):
        plt.subplot(n_rows, n_cols,4+i_model)
        plt.imshow(Y_soft[i_model,i_test,:,64,:], cmap='gray')
    
    plt.subplot(n_rows, n_cols,9)
    plt.imshow(KW[:,64,:])
    
    plt.subplot(n_rows, n_cols,10)
    plt.imshow(boundary_all[:,64,:])
    
    fig.savefig(uncertainty_thumbs + 'X_' + str(i_test) + '_sagittal.png')
    plt.close(fig)
    
    ######
    fig, ax = plt.subplots(figsize=(16,10), nrows= n_rows, ncols= n_cols )
    
    plt.subplot(n_rows, n_cols,1)
    plt.imshow(X_test[i_test,64,:,:,0], cmap='gray')
    plt.subplot(n_rows, n_cols,2)
    plt.imshow(Y_test[i_test,64,:,:,1], cmap='gray')
    plt.subplot(n_rows, n_cols,3)
    plt.imshow(Y_boundary[i_test,64,:,:], cmap='gray')
    
    for i_model in range(n_model):
        plt.subplot(n_rows, n_cols,4+i_model)
        plt.imshow(Y_soft[i_model,i_test,64,:,:], cmap='gray')
    
    plt.subplot(n_rows, n_cols,9)
    plt.imshow(KW[64,:,:])
    
    plt.subplot(n_rows, n_cols,10)
    plt.imshow(boundary_all[64,:,:])
    
    fig.savefig(uncertainty_thumbs + 'X_' + str(i_test) + '_coronal.png')
    plt.close(fig)
    
    





'''






































