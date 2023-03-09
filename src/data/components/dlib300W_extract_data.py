# import math
import cv2
import os
# import random 
import numpy as np
from PIL import Image
# import imutils
import matplotlib.pyplot as plt
from math import *
import random
import xml.etree.ElementTree as ET 
# import torch
import torchvision.transforms.functional as TF
# from torchvision import datasets, models, transforms
from torch.utils.data import Dataset

class FaceLandmarksDataset(Dataset):

    def __init__(
                self, 
                data_path = None,
                root_dir = None,
                transform = None):
        
        super().__init__()
        self.data_path = data_path
        tree = ET.parse(data_path)
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = root_dir

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])

                landmark.append([x_coordinate, y_coordinate])
                    
            self.landmarks.append(landmark)
                    
        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])
        landmarks = landmarks - 0.5
        return image, landmarks
    

# sandbox





# tree = ET.parse('data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
# root = tree.getroot()

# print(root[2][0][0].attrib)
