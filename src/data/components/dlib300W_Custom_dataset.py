import os
import numpy as np
from PIL import Image, ImageDraw
from math import *
import xml.etree.ElementTree as ET 
import torch
import torchvision
from torch.utils.data import Dataset
from typing import Optional
from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2



class DlibDataset(Dataset):
    def __init__(self, 
                dataset_path: str =  None,
                root_dir : str =  None,
                ):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.root_dir = root_dir
        
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
        
        top, left, bottom, right = self.extract_bbox(index)
        original_image : Image = Image.open(self.img_filenames[index]).convert('RGB')
        
        croped_image : Image = original_image.crop((left, top, right, bottom))
        landmark = self.landmarks[index]
        landmark -= np.array([left, top])
        return croped_image, landmark
        
    
    def annotate_image(image: Image, landmark: np.ndarray) ->Image:
        draw = ImageDraw.Draw(image)
        for i in range(landmark.shape[0]):
            draw.ellipse((landmark[i][0] - 2, landmark[i][1] -2,
                          landmark[i][0] + 2, landmark[i][1] +2), fill = (255, 0, 0))
        return image
    
    
    
    
    
class TransformDataset(Dataset):
    def __init__(self, dataset:DlibDataset, transform: Optional[Compose] = None ):
        self.dataset = dataset
        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(224,224),
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=45, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image, landmark = self.dataset[index]
        image = np.array(image)
        transformed =  self.transform(
            image = image, keypoints = landmark
        )
        image, landmark = transformed["image"], transformed["keypoints"]
        _, height, width = image.shape 
        landmark = landmark / np.array([width, height]) -0.5
        return image, landmark.astype(np.float32)
    
    @staticmethod
    def annotate_tensor(image: torch.tensor, landmark: np.ndarray) -> Image:
        IMG_MEAN = [0.485, 0.456, 0.406]
        IMG_STD = [0.229, 0.224, 0.225]
        
        def denormalize(x, mean = IMG_MEAN, std = IMG_STD) ->torch.Tensor:
            #3, H, W, B
            
            ten = x.clone().permute(1, 2, 3, 0)
            #'.clone()' make a copy of original image
            for t, m, s in zip(ten, mean, std):
                t.mul_(s).add_(m)
            #B, 3, H, W
            #clamp here like clip in np 
            return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)
        
        image_denormalized = denormalize(image)
        images_to_save = []
        for lm, img in zip(landmark, image_denormalized):
            img = img.permute(1, 2, 0).numpy()*255
            h, w, _ = img.shape
            lm = (lm + 0.5) * np.array([w, h]) #convert to image pixel coordinate
            
            img = DlibDataset.annotate_image(Image.fromarray(img.astype(np.uint8)), lm)
            images_to_save.append(torchvision.transforms.ToTensor()(img))
            
        return torch.stack(images_to_save)
                
# dataset = DlibDataset()
# dataset = TransformDataset(dataset)
    
# cropped_img, landmark = dataset[0]
# print(landmark)
# plt.imshow(cropped_img)
# plt.show()

# image_annotated = DlibDataset.annotate_image(cropped_img, landmark)
# plt.imshow(image_annotated)
# plt.show()
