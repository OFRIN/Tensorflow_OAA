
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

        self.count_dic = {}
        
        if not os.path.isdir(self.oaa_dir):
            os.makedirs(self.oaa_dir)
    
    def normalize(self, class_map):
        max_value = np.max(class_map)
        class_map = class_map / (max_value + 1e-8) * 255
        return class_map.astype(np.uint8)
    
    def update(self, image_paths, predictions, labels, attention_maps):
        
        condition = np.logical_and(predictions >= self.oaa_threshold, labels == 1)
        correct_indices = np.arange(len(condition))[np.sum(condition, axis = -1) > 0]
        
        image_paths = image_paths[correct_indices]
        attention_maps = attention_maps[correct_indices]
        predictions = predictions[correct_indices]
        corrects = condition[correct_indices]

        for image_path, attention_map, prediction, correct in zip(image_paths, attention_maps, predictions, corrects):
            image_path = image_path.decode('utf-8')
            oaa_image_name = image_path.replace(self.root_dir, '')

            for class_index in range(self.classes):
                if not correct[class_index]:
                    continue
                
                class_attention_map = self.normalize(attention_map[..., class_index])
                # class_attention_map = cv2.resize(class_attention_map, (224, 224))
                
                try:
                    self.count_dic[oaa_image_name] += 1
                except KeyError:
                    self.count_dic[oaa_image_name] = 1
                
                oaa_npy_path = self.oaa_dir + oaa_image_name.replace('.jpg', '_{}.png'.format(self.class_names[class_index]))
                
                oaa_image_dir = os.path.dirname(oaa_npy_path)
                if not os.path.isdir(oaa_image_dir):
                    os.makedirs(oaa_image_dir)
                
                if os.path.isfile(oaa_npy_path):
                    attention_maps = list(np.load(oaa_npy_path, allow_pickle = True))
                    attention_maps.append([class_attention_map, prediction[class_index]])
                else:
                    attention_maps = [ 
                        [class_attention_map, prediction[class_index]] 
                    ]

                np.save(oaa_npy_path, attention_maps)
