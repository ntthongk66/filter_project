from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
# from torchvision.transforms import transforms
# from albumentations.pytorch import ToTensorV2
# import albumentations as A
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

class DLIB300WDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (5000, 1000, 666),
        batch_size = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        
        super().__init__()
        self.save_hyperparameters(logger = False)
        
        
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
    def prepare_data(self):
        pass
        
        
    def setup(self, stage: Optional[str] = None):
        
        class Transforms():
            def __init__(self):
                pass

            def crop_face(self, image, landmarks, crops):
                top = int(crops['top'])
                left = int(crops['left'])
                height = int(crops['height'])
                width = int(crops['width'])
                
                image = TF.crop(image, top, left, height, width)
                img_shape = np.array(image).shape

                landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
                landmarks = landmarks/torch.tensor([img_shape[1], img_shape[0]])

                return image, landmarks

            def resize(self, image, landmarks, img_size):
                image = TF.resize(image, img_size)

                return image, landmarks

            def color_jitter(self, image, landmarks):
                color_jitter = transforms.ColorJitter(brightness = random.random(),
                                                    contrast = random.random(),
                                                    saturation= random.random(),
                                                    hue = random.uniform(0,0.5))
                image =  color_jitter(image)
                return image, landmarks

            def rotate(self, image, landmarks, angle):
                angle = random.uniform(-angle, +angle)

                transformation_matrix = torch.tensor([
                    [+cos(radians(angle)), -sin(radians(angle))],
                    [+sin(radians(angle)), +cos(radians(angle))],
                    
                ])

                image = imutils.rotate(np.array(image), angle)

                landmarks = landmarks -0.5
                new_landmarks = np.matmul(landmarks, transformation_matrix)
                new_landmarks = new_landmarks +0.5
                return  Image.fromarray(image), new_landmarks

            def __call__ (self, image, landmarks, crops):
                image = Image.fromarray(image)
                
                # aumentation
                image, landmarks = self.crop_face(image, landmarks, crops)
                image, landmarks = self.resize(image, landmarks, (224, 224))
                image, landmarks = self.color_jitter(image, landmarks)
                image, landmarks = self.rotate(image, landmarks, angle = random.randint(-20, 20))

                image = TF.to_tensor(image)
                image = TF.normalize(image, [0.5], [0.5])

                return image, landmarks
            
            
        class FaceLandmarksDataset(Dataset):

            def __init__(self, transform = None):
                tree = ET.parse('data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
                root = tree.getroot()

                self.image_filenames = []
                self.landmarks = []
                self.crops = []
                self.transform = transform
                self.root_dir = 'data/ibug_300W_large_face_landmark_dataset'

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
        
        dataset = FaceLandmarksDataset(Transforms())
        
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset= dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
            
        
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def draw_batch(self):
        images, landmarks = next(iter(self.train_dataloader()))
        batch_size = len(images)
        grid_size = math.sqrt(batch_size)
        grid_size = math.ceil(grid_size)
        fig = plt.figure(figsize= (10,10))
        fig.subplots_adjust(left = 0, right = 1, bottom = 0, top = 1, hspace = 0.05, wspace = 0.05)
        for i in range(batch_size):
            ax = fig.add_subplot(grid_size, grid_size, i+1, xticks=[], yticks=[])
            # plot_keypoint_image(images[i], landmarks[i])
            
            landmarks[i] = (landmarks[i] + 0.5 ) * 224
            # plt.figure(figsize=(10 ,10))
            plt.imshow(images[i].numpy().squeeze(), cmap = 'gray');
            kpt = []
            for j in range(68):
                kpt.append(plt.plot(landmarks[i][j][0], landmarks[i][j][1], 'go'))
        plt.show()
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    _ = DLIB300WDataModule()

