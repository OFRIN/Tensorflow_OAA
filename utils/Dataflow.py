# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import copy
import time
import random

import multiprocessing as mp

import numpy as np

from tensorpack import imgaug, dataset
from tensorpack.dataflow import AugmentImageComponent, PrefetchData, BatchData, MultiThreadMapData, RNGDataFlow

class DataFlow(RNGDataFlow):
    def __init__(self, dataset, option):
        self.option = option
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        indexs = np.arange(len(self.dataset))
        if self.option['shuffle']:
            self.rng.shuffle(indexs)
        
        for i in indexs:
            image_path, label = self.dataset[i]
            
            image = cv2.imread(image_path)
            if image is None:
                print('[!] Error = {}'.format(image_path))
                continue
            
            image = cv2.resize(image, self.option['image_size'], interpolation = cv2.INTER_CUBIC)
            
            if self.option['OAA']:
                yield [image.astype(np.float32), np.asarray(label, dtype = np.float32), image_path]
            else:
                yield [image.astype(np.float32), np.asarray(label, dtype = np.float32)]

def generate_dataflow(dataset, option):
    if option['number_of_cores'] == -1:
        option['number_of_cores'] = mp.cpu_count()
    
    ds = DataFlow(dataset, option)
    ds = AugmentImageComponent(ds, option['augmentors'], copy = False)
    
    if option['number_of_cores'] < 16:
        print('[!} Warning = DataFlow may become the bottleneck when too few processes are used.')
    
    ds = PrefetchData(ds, option['num_prefetch_for_dataset'], option['number_of_cores'])

    ds = BatchData(ds, option['batch_size'], remainder = option['remainder'])
    ds = PrefetchData(ds, option['num_prefetch_for_batch'], 2)
    
    return ds

