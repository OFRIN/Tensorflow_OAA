# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import cv2
import sys
import copy
import time
import glob
import json
import random
import argparse

import numpy as np
import tensorflow as tf

import multiprocessing as mp

from core.Config_OAA import *
from core.Classifier import *

from core.Online_Attention_Accumulation import *

from utils.Utils import *
from utils.Timer import *
from utils.Dataflow import *
from utils.Generator import *
from utils.Tensorflow_Utils import *

if __name__ == '__main__':
    #######################################################################################
    # 0. Config
    #######################################################################################
    flags = get_config()
    flags.warmup_iteration = int(flags.max_iteration * 0.05) # warmup iteration = 5%

    num_gpu = len(flags.use_gpu.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.use_gpu

    flags.batch_size = flags.batch_size_per_gpu * num_gpu
    if flags.batch_size > 256:
        flags.init_learning_rate *= flags.batch_size / 256
        flags.alpha_learning_rate *= flags.batch_size / 256

    model_name = 'VGG16-{}-{}'.format(flags.experimenter, get_today())
    model_dir = './experiments/model/{}/'.format(model_name)
    tensorboard_dir = './experiments/tensorboard/{}/'.format(model_name)

    ckpt_format = model_dir + '{}.ckpt'
    log_txt_path = model_dir + 'log.txt'

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if os.path.isfile(log_txt_path):
        open(log_txt_path, 'w').close()

    #######################################################################################
    # 1. Dataset
    #######################################################################################
    log_print('# {}'.format(model_name), log_txt_path)

    train_dir = flags.root_dir + '/train/'
    class_names = sorted(os.listdir(train_dir))
    classes = len(class_names)

    train_dic = {}
    for class_index, class_name in enumerate(class_names):
        label = single_one_hot(class_index, classes)
        train_dic[class_name] = [[image_path, label] for image_path in glob.glob(train_dir + class_name + '/*.jpg')]
    
    #######################################################################################
    # 1.1. Info (Dataset)
    #######################################################################################
    log_print('\n', log_txt_path)
    log_print('### Train', log_txt_path)
    for key in class_names:
        log_print('=> {:10s} = {}'.format(key, len(train_dic[key])), log_txt_path)
    
    #######################################################################################
    # 2. Generator & Queue
    #######################################################################################
    dataflow_option = {
        'OAA' : True,
        
        'augmentors' : [],

        'shuffle' : True,
        'remainder' : False,
        
        'batch_size' : flags.batch_size // num_gpu,
        'image_size' : (flags.image_size, flags.image_size),
        
        'num_prefetch_for_dataset' : 10,
        'num_prefetch_for_batch' : 5,
        
        'number_of_cores' : 2,
    }

    train_image_var = tf.placeholder(tf.float32, [None, flags.image_size, flags.image_size, 3])
    train_label_var = tf.placeholder(tf.float32, [None, len(class_names)])
    train_image_paths_var = tf.placeholder(tf.string, [None])
    is_training = tf.placeholder(tf.bool)
    
    generator_func = lambda ds: Generator({
        'dataset' : ds, 
        'placeholders' : [train_image_var, train_label_var, train_image_paths_var], 

        'queue_size' : 10, 
        'batch_size' : flags.batch_size // num_gpu,
    })

    dataset = []
    for class_name in class_names:
        dataset += train_dic[class_name]
    
    if flags.OAA_update_iteration == -1:
        flags.OAA_update_iteration = len(dataset) // flags.batch_size
        log_print('[i] calculate 1 epoch = {} iteration'.format(flags.OAA_update_iteration), log_txt_path)    

    train_dataset_list = [generate_dataflow(dataset, dataflow_option) for _ in range(num_gpu)]
    train_generators = [generator_func(train_dataset_list[i]) for i in range(num_gpu)]
    
    log_print('[i] generate dataset and generators', log_txt_path)
    log_print('{}'.format(json.dumps(flags_to_dict(flags), indent='\t')), log_txt_path)
    
    #######################################################################################
    # 3. Model
    #######################################################################################
    oaa_updater = Online_Attention_Accumulation({
        'OAA_threshold' : flags.OAA_threshold,
        
        'OAA_dir' : flags.OAA_dir,
        'root_dir' : flags.root_dir,
        
        'class_names' : class_names,
    })
    
    logits_list = []
    predictions_list = []
    attention_maps_list = []

    train_label_ops = []
    train_image_paths_ops = []

    model_option = {
        'classes' : len(class_names),
        'log_txt_path' : log_txt_path,
    }
    
    for gpu_id in range(num_gpu):
        train_image_op, train_label_op, train_image_paths_op = train_generators[gpu_id].dequeue()

        train_label_ops.append(train_label_op)
        train_image_paths_ops.append(train_image_paths_op)
        
        with tf.device(tf.DeviceSpec(device_type = "GPU", device_index = gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse = gpu_id > 0):
                output_dic = Classifier(train_image_op, is_training, model_option)

                logits_list.append(output_dic['logits'])
                predictions_list.append(output_dic['predictions'])
                attention_maps_list.append(output_dic['attention_maps'])

        log_print('[i] build model (gpu = %d)'%gpu_id, log_txt_path)

    logits_op = tf.concat(logits_list, axis = 0)
    predictions_op = tf.concat(predictions_list, axis = 0)
    attention_maps_op = tf.concat(attention_maps_list, axis = 0)

    label_op = tf.concat(train_label_ops, axis = 0)
    image_paths_op = tf.concat(train_image_paths_ops, axis = 0)

    print_ops = [logits_op, predictions_op, attention_maps_op, label_op, image_paths_op]
    log_print('[i] finish concatenation {}, {}, {}, {}, {}'.format(*print_ops), log_txt_path)

    class_loss_op = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_op, labels = label_op)
    class_loss_op = tf.reduce_mean(class_loss_op)

    train_vars = tf.trainable_variables()
    l2_vars = [var for var in train_vars if 'kernel' in var.name or 'weights' in var.name]
    l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in l2_vars]) * flags.weight_decay

    loss_op = class_loss_op + l2_reg_loss_op

    log_print('[i] finish optimizer', log_txt_path)
    
    #######################################################################################
    # 4. optimizer
    #######################################################################################
    global_step = tf.placeholder(dtype = tf.int32)

    warmup_lr_op = tf.to_float(global_step) / tf.to_float(flags.warmup_iteration) * flags.init_learning_rate
    decay_lr_op = tf.train.cosine_decay(
        flags.init_learning_rate,
        global_step = global_step - flags.warmup_iteration,
        decay_steps = flags.max_iteration - flags.warmup_iteration,
        alpha = flags.alpha_learning_rate
    )

    learning_rate = tf.where(global_step < flags.warmup_iteration, warmup_lr_op, decay_lr_op)

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.MomentumOptimizer(learning_rate, momentum = 0.9, use_nesterov = True).minimize(loss_op, colocate_gradients_with_ops = True)
        
    #######################################################################################
    # 5. Metrics
    #######################################################################################
    correct_op = tf.equal(tf.greater_equal(predictions_op, 0.5), tf.greater_equal(label_op, 0.5))
    accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100
    
    #######################################################################################
    # 6. tensorboard
    #######################################################################################
    train_summary_dic = {
        'Loss/Total_Loss' : loss_op,
        'Loss/Clasification_Loss' : class_loss_op,
        'Loss/L2_Regularization_Loss' : l2_reg_loss_op, 
        'Accuracy/Train_Accuracy' : accuracy_op,
        'Learning_rate' : learning_rate,
    }
    train_summary_op = tf.summary.merge([tf.summary.scalar(name, train_summary_dic[name]) for name in train_summary_dic.keys()])

    valid_summary_dic = {
        'Accuracy/Validation_Accuracy' : tf.placeholder(tf.float32),
        'Accuracy/Validation_Positive_Accuracy' : tf.placeholder(tf.float32),
        'Accuracy/Validation_Negative_Accuracy' : tf.placeholder(tf.float32),
    }
    valid_summary_op = tf.summary.merge([tf.summary.scalar(name, valid_summary_dic[name]) for name in valid_summary_dic.keys()])

    train_writer = tf.summary.FileWriter(tensorboard_dir)
    log_print('[i] tensorboard directory is {}'.format(tensorboard_dir), log_txt_path)

    #######################################################################################
    # 7. create Session and Saver.
    #######################################################################################
    sess = tf.Session()
    coord = tf.train.Coordinator()

    saver = tf.train.Saver(
        # var_list = tf.trainable_variables(scope = 'tower0'),
        max_to_keep = 20
    )
    
    # pretrained model
    pretrained_model_name = 'vgg_16'
    pretrained_model_path = './pretrained_models/{}/model.ckpt'.format(pretrained_model_name)

    imagenet_saver = tf.train.Saver(var_list = [var for var in train_vars if pretrained_model_name in var.name])
    imagenet_saver.restore(sess, pretrained_model_path)

    log_print('[i] restore pretrained model ({}) -> {}'.format(pretrained_model_name, pretrained_model_path), log_txt_path)

    #######################################################################################
    # 8. initialize
    #######################################################################################
    sess.run(tf.global_variables_initializer())

    for train_generator in train_generators:
        train_generator.set_session(sess)
        train_generator.set_coordinator(coord)
        train_generator.start()

        log_print('[i] start train generator ({})'.format(train_generator), log_txt_path)

    #######################################################################################
    # 9. Train
    #######################################################################################
    loss_list = []
    class_loss_list = []
    l2_reg_loss_list = []
    accuracy_list = []
    train_time = time.time()

    oaa_time = 0
    oaa_time_list = []

    oaa_ops = [image_paths_op, predictions_op, label_op, attention_maps_op]
    train_ops = [train_op, loss_op, class_loss_op, l2_reg_loss_op, accuracy_op, train_summary_op]

    oaa_timer = Timer()
    
    for iter in range(1, flags.max_iteration + 1):
        _feed_dict = {
            is_training : True,
            global_step : iter,
        }
        data = sess.run(train_ops + oaa_ops, feed_dict = _feed_dict)
        _, loss, class_loss, l2_reg_loss, accuracy, summary = data[:len(train_ops)]

        loss_list.append(loss)
        class_loss_list.append(class_loss)
        l2_reg_loss_list.append(l2_reg_loss)
        accuracy_list.append(accuracy)
        train_writer.add_summary(summary, iter)
        
        if iter >= flags.OAA_update_iteration:
            # update OAA - time
            oaa_timer.tik()
            oaa_updater.update(*data[len(train_ops):])
            oaa_time_list.append(oaa_timer.tok())

            # log_print('[i] iter = {}, oaa update = {}ms'.format(iter, oaa_timer.tok()), log_txt_path)
        
        if iter % flags.log_iteration == 0:
            loss = np.mean(loss_list)
            class_loss = np.mean(class_loss_list)
            l2_reg_loss = np.mean(l2_reg_loss_list)
            accuracy = np.mean(accuracy_list)
            train_time = int(time.time() - train_time)

            if len(oaa_time_list) > 0:
                oaa_time = np.sum(oaa_time_list)
            
            log_print('[i] iter = {}, loss = {:.4f}, class_loss = {:.4f}, l2_reg_loss = {:.4f}, accuracy = {:.2f}, oaa_time = {}ms, train_time = {}sec'.format(iter, loss, class_loss, l2_reg_loss, accuracy, oaa_time, train_time), log_txt_path)
            # log_print('[i] queue_sizes = {}'.format(list([train_generator.size() for train_generator in train_generators])), log_txt_path)
            
            loss_list = []
            class_loss_list = []
            l2_reg_loss_list = []
            accuracy_list = []
            train_time = time.time()

            oaa_time_list = []
        
        #######################################################################################
        # 10. Validation
        #######################################################################################
        if iter % flags.valid_iteration == 0:
            saver.save(sess, ckpt_format.format(iter))   
            