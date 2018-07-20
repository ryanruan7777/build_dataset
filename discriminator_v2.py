# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 14:42:09 2018

@author: Ryan
"""
import tensorflow as tf
from ops import *
from utils import *
'''
ops and utils all from pix2pix program
a encoder-decoder discrimintor
'''

def discriminator(self, image, y = None, reuse = False):
    #shape对不上 ！！！！
    with tf.variable_scope("discriminator") as scope:
        
        s = self.output_size
        s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)
        #image is 256x256
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
            
        # a autoencoder to create a image
        # encoder:
        h0 = lrelu(conv2d(image, self.df_dim, name = 'h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name = 'h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name = 'h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name = 'h3_conv')))#16*16
        
        #decoder:
        self.x1, self.x1_w, self.x1_b = deconv2d(tf.nn.relu(h3), [self.batch_size, s8, s8, self.gf_dim*8], name ='d_x1',with_w = True)
        print(self.x1.shape)
        x1 = tf.nn.dropout(self.d_bn_x1(self.x1), 0.5)
       
        
        self.x2, self.x2_w, self.x2_b = deconv2d(tf.nn.relu(x1), [self.batch_size, s4, s4, self.gf_dim*4], name ='d_x2',with_w = True)
        x2 = tf.nn.dropout(self.d_bn_x2(self.x2), 0.5)
        print(self.x2.shape)
        
        
        self.x3, self.x3_w, self.x3_b = deconv2d(tf.nn.relu(x2), [self.batch_size, s2, s2, self.gf_dim*2], name ='d_x3',with_w = True)
        x3 = self.d_bn_x3(self.x3)
        print(self.x3.shape)
        
        
        self.x4, self.x4_w, self.x4_b = deconv2d(tf.nn.relu(x3), [self.batch_size, s, s, self.output_c_dim], name ='d_x4',with_w = True)
        x4 = self.d_bn_x4(self.x4)
        print(self.x4.shape)
                   
        return tf.nn.sigmoid(x4), x4

def loss():
    
    d_loss