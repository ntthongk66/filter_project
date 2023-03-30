import math
import cv2
import os
import random 
import numpy as np
from PIL import Image
import imutils
import matplotlib.pyplot as plt
from math import *
import random
import xml.etree.ElementTree as ET 
import torch
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2



class CustomDlibData(Dataset):
    def __init__(self, 
                dataset_path = "data\ibug_300W_large_face_landmark_dataset\labels_ibug_300W_train.xml",
                root_dir  = "data\ibug_300W_large_face_landmark_dataset",
                # dataset_path,
                # root_dir,
                kpt_transform =None,
                img_transform = None):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.root_dir = root_dir
        self.img_transform = img_transform

        
        self.img_filenames = []
        self.crops = []
        self.landmarks = []
        
        tree = ET.parse(dataset_path)
        root = tree.getroot()
        
        '''
        construct in root
        root[2] : list of [file = 'abc.jpg', with = 123, height = 123  ]
        
        
        
        '''
        
        for filename in root[2]:
            self.img_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
            
            self. crops.append(filename[0].attrib)
            
            landmark = []
            for num in range(68):
                x_codi = int(filename[0][num].attrib['x'])
                y_codi = int(filename[0][num].attrib['y'])
                
                #! "codi" stand for coordinate
                
                landmark.append([x_codi, y_codi])
            
            self.landmarks.append(landmark)
            
        self.landmarks = np.array(self.landmarks).astype('float32')
            
            # todo: check len img_file and num of landmarks
        assert len(self.img_filenames) == len(self.landmarks)
            
    def __len__(self):
        return len(self.img_filenames)
    
    def extract_bbox(self, index):
        top = int(self.crops[index]['top'])
        left = int(self.crops[index]['left'])
        height = int(self.crops[index]['height'])
        width = int(self.crops[index]['width'])
        
        bottom = top + height
        right = left + width
        
        return top, left, bottom, right
    
    def __getitem__(self, index): 
        img = cv2.imread(self.img_filenames[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        top, left, bottom, right = self.extract_bbox(index)
        img_shape = np.array(img).shape
        height = img_shape[0]
        width = img_shape[1]
        
        
        top = max(0, top) #y_min
        left = max(0, left) #x_min
        bottom = min(bottom, height) #y_max
        right = min (right, width)  #x_max
        
        img_size = 224
        landmark = self.landmarks[index]
        
        if True:
            transform = A.Compose([
                A.Crop(left, top, right, bottom),
                A.Resize(img_size, img_size),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
            , keypoint_params= A.KeypointParams(format= 'xy', remove_invisible= False, angle_in_degrees = True )    )
            
            transformed = transform(image = img, keypoints = landmark )
            img = transformed['image']
            landmark = transformed['keypoints']
            landmark = np.array(landmark).astype('float32')
            
            img = np.clip(img, 0, 1)
            # rescale img to [0...1]
            
        return img, landmark
    
    
dataset = CustomDlibData()
    
_, landmark = dataset[0]
print()
# def visualize(samples = 32):
#     grid_size = math.sqrt(samples)
#     grid_size = math.ceil(grid_size)
#     fig = plt.figure(figsize=(10, 10))
#     fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
#     for i in range(samples):
#         ax = fig.add_subplot(grid_size, grid_size, i+1, xticks=[], yticks=[])
#         image, landmark = dataset[i]
#         image = image.squeeze().permute(1,2,0)
#         plt.imshow(image)
#         kpt = []
#         for j in range(68):
#             kpt.append(plt.plot(landmark[j][0], landmark[j][1], 'g.'))
#     plt.tight_layout()
#     plt.show()

# visualize()