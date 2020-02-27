# Copyright (C) 2019 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import numpy as np
import tensorflow as tf

import core.vgg_16.vgg_16 as vgg16

from utils.Utils import *

def get_attention_maps(feature_maps):
    attention_maps = tf.nn.relu(feature_maps)
    
    # max_value = tf.math.reduce_max(heatmaps, axis = [1, 2])
    # heatmaps = heatmaps - min_value) / (max_value - min_value) * 255.
    
    return tf.identity(attention_maps, name = 'attention_maps')

def Classifier(x, is_training, option):
    x = x[..., ::-1] - [vgg16._R_MEAN, vgg16._G_MEAN, vgg16._B_MEAN]
    log_print('[i] VGG16 mean = {}'.format([vgg16._R_MEAN, vgg16._G_MEAN, vgg16._B_MEAN]), option['log_txt_path'])
    
    with tf.contrib.slim.arg_scope(vgg16.vgg_arg_scope()):
        x = vgg16.vgg_16(x, num_classes=1000, is_training=is_training, dropout_keep_prob=0.5)
        
        log_print('[i] vgg16 feature_maps = {}'.format(x), option['log_txt_path'])
    
    with tf.variable_scope('Classifier', reuse = tf.AUTO_REUSE, custom_getter = None):
        # feature extractor
        channels = x.get_shape().as_list()[-1]
        x = tf.layers.conv2d(x, channels, (3, 3), padding = 'same', activation = tf.nn.relu)
        x = tf.layers.conv2d(x, channels, (3, 3), padding = 'same', activation = tf.nn.relu)
        x = tf.layers.conv2d(x, channels, (3, 3), padding = 'same', activation = tf.nn.relu)
        feature_maps = tf.layers.conv2d(x, option['classes'], [1, 1], 1, name = 'feature_maps')
        attention_maps = get_attention_maps(feature_maps)

        log_print('[i] attention_maps = {}'.format(attention_maps), option['log_txt_path'])
        
        logits = tf.reduce_mean(feature_maps, axis = [1, 2], name = 'GAP')
        predictions = tf.nn.sigmoid(logits, name = 'outputs')
    
    return {
        'logits' : logits,
        'predictions' : predictions,
        'feature_maps' : feature_maps,
        'attention_maps' : attention_maps,
    }

