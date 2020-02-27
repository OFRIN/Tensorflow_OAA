import os
import cv2
import glob

import numpy as np
import tensorflow as tf

from core.Config_OAA import *
from core.Classifier import *

# 1. dataset
flags = get_config()

train_dir = flags.root_dir + '/train/'
class_names = sorted(os.listdir(train_dir))
classes = len(class_names)

train_dic = {}
for class_index, class_name in enumerate(class_names):
    label = single_one_hot(class_index, classes)
    train_dic[class_name] = [[image_path, label] for image_path in glob.glob(train_dir + class_name + '/*.jpg')]

# 2. Classifier
image_var = tf.placeholder(tf.float32, [None, flags.image_size, flags.image_size, 3])

output_dic = Classifier(image_var, False, {
    'classes' : len(class_names),
    'log_txt_path' : None,
})

predictions_op = output_dic['predictions']
attention_maps_op = output_dic['attention_maps']

# 3. 
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
saver.restore(sess, './experiments/model/VGG16-JSH-2020-02-26-14h50m17s/20000.ckpt')

oaa_dir = './dataset/OAA/'
result_tag = '_OAA.jpg'

def merge(images):
    length = len(images)
    h, w, c = images[0].shape

    merge_image = np.zeros((h, w * length, c), dtype = np.uint8)
    for i in range(length):
        merge_image[:, i * w : (i + 1) * w, :] = images[i]

    return merge_image

def set_demo_image(demo_image):
    demo_image = cv2.resize(demo_image, (flags.image_size, flags.image_size))
    demo_image = cv2.applyColorMap(demo_image, cv2.COLORMAP_JET)
    return demo_image

def set_text(demo_image, text):
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    cv2.rectangle(demo_image, (0, 0), (text_size[0], text_size[1] + 5), (0, 255, 0), cv2.FILLED)
    cv2.putText(demo_image, text, (0, text_size[1]), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
    return demo_image

for image_path, label in train_dic['daisy']:
    image_name = os.path.basename(image_path).replace('.jpg', '')

    image = cv2.imread(image_path)
    image = cv2.resize(image, (flags.image_size, flags.image_size))
    
    oaa_image_name = image_path.replace(flags.root_dir, '').replace('.jpg', '_daisy.npy') # history
    oaa_image_path = oaa_dir + oaa_image_name

    preds, attention_maps = sess.run([predictions_op, attention_maps_op], feed_dict = {image_var : [image.astype(np.float32)]})

    pred = preds[0]
    attention_map = attention_maps[0]

    print(pred, label)
    
    cam = set_demo_image(attention_map[..., 0].astype(np.uint8))
    cam = cv2.addWeighted(image, 0.5, cam, 0.5, 0.0)
    cam = set_text(cam, '# CAM')

    oaa = set_demo_image(cv2.imread(oaa_image_path))
    oaa = cv2.addWeighted(image, 0.5, oaa, 0.5, 0.0)
    oaa = set_text(oaa, '# OAA')
    
    merge_image = merge([image, cam, oaa])

    cv2.imshow('show_image', merge_image)
    cv2.waitKey(0)

    cv2.imwrite('./results/' + image_name + result_tag, merge_image)