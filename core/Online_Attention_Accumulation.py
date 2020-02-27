
import os
import cv2
import numpy as np

class Online_Attention_Accumulation:
    def __init__(self, option):
        self.oaa_threshold = option['OAA_threshold']

        self.oaa_dir = str(option['OAA_dir'])
        self.root_dir = str(option['root_dir'])

        self.classes = len(option['class_names'])
        self.class_names = option['class_names']
        
        if not os.path.isdir(self.oaa_dir):
            os.makedirs(self.oaa_dir)
    
    def update(self, image_paths, predictions, labels, attention_maps):
        
        condition = np.logical_and(predictions >= self.oaa_threshold, labels == 1)
        correct_indices = np.arange(len(condition))[np.sum(condition, axis = -1) > 0]
        
        image_paths = image_paths[correct_indices]
        attention_maps = attention_maps[correct_indices]
        corrects = condition[correct_indices]

        for image_path, attention_map, correct in zip(image_paths, attention_maps, corrects):
            image_path = image_path.decode('utf-8')
            oaa_image_name = image_path.replace(self.root_dir, '')

            for class_index in range(self.classes):
                if not correct[class_index]:
                    continue
                
                class_attention_map = attention_map[..., class_index].astype(np.uint8)
                
                oaa_image_path = self.oaa_dir + oaa_image_name.replace('.jpg', '_{}.png'.format(self.class_names[class_index]))
                # print(oaa_image_path)
                
                oaa_image_dir = os.path.dirname(oaa_image_path)
                if not os.path.isdir(oaa_image_dir):
                    os.makedirs(oaa_image_dir)
                
                if os.path.isfile(oaa_image_path):
                    prior_attention_map = cv2.imread(oaa_image_path)
                    prior_attention_map = cv2.cvtColor(prior_attention_map, cv2.COLOR_BGR2GRAY)
                    class_attention_map = np.maximum(class_attention_map, prior_attention_map)

                cv2.imwrite(oaa_image_path, class_attention_map)
