import cv2
import imutils
import os
import math
from facenet_pytorch import MTCNN
import torch
from PIL import Image, ImageDraw
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
import pyrootutils

# __file__ = '/Users/tiendzung/Downloads/facial_landmarks-wandb/notebooks/explore_dlib.ipynb'
path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root")
config_path = str(path / "configs" / "model")
output_path = path / "outputs"
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)



# device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
device = torch.device('cuda')
print(device)
mtcnn = MTCNN(thresholds= [0.7, 0.7, 0.8] ,keep_all=True, device = device)

transform = Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

from src.models.components.simple_resnet import SimpleResnet
from src.models.dlib300w_module import DlibLiModule
#!paste glass to image
#* sung glass
sung_glass = cv2.imread('E:/filter_project/filter_project/sunglasses.png', cv2.IMREAD_UNCHANGED)


def paste_to_img( frame_bgr  ,top_left, top_right, sung_glass_img = sung_glass):
    
    height, width = frame_bgr.shape[:2]
    bg_bgr = frame_bgr[:,:,0:3]
    bg_mask = frame_bgr[:,:,2]
    
    # ori_h, ori_w, _ = sung_glass_img.shape
    #! rotate part
    
    degree =   math.atan2((top_left[1]-top_right[1]),(top_left[0]-top_right[0]))
    degree = int(degree*360/(2*math.pi))
    sung_glass_img = imutils.rotate_bound(sung_glass_img, degree -180 )
    print(degree)
    ori_h, ori_w, _ = sung_glass_img.shape
    virtual_width = abs(top_right[0]-top_left[0]) 
    
    r = virtual_width/ori_w
    dim = (int(virtual_width), int(ori_h*r) )
    resized_sung_glass = cv2.resize(sung_glass_img, dim, interpolation=cv2.INTER_AREA)
    h_filter, w_filter, _= resized_sung_glass.shape
    
    x = int(top_left[0])
    y = int(top_left[1])
    w =  w_filter
    h = h_filter
    
    
    bgr = resized_sung_glass[:,:,0:3]
    mask = resized_sung_glass[:, :, 3]
        
    bgr_new = bg_bgr.copy()
    if  ((bgr.shape[0] != h or bgr.shape[1] != w)) :
        
        print(type(bgr.shape[0]), type(h), type(bgr.shape[1]), type(w))
        print((bgr.shape[0] == h & bgr.shape[1] == w))
        return frame_bgr
    
    else :
        bgr_new[y:y+h, x:x+w] = bgr
        mask_new = np.zeros((height, width), dtype = np.uint8)
        mask_new[y:y+h, x:x+w] = mask
                
        mask_combined = cv2.multiply(bg_mask, mask_new)
        mask_combined = cv2.cvtColor(mask_combined, cv2.COLOR_GRAY2BGR)
                
        result = np.where(mask_combined==255, bgr_new, bg_bgr)
        print('glass')
                # frame_bgr = result
                # bg_bgr = frame_bgr[:,:,0:3]
        return result


model = DlibLiModule.load_from_checkpoint(checkpoint_path='E:/filter_project/filter_project/logs/train/runs/2023-04-08_00-04-27/checkpoints/epoch_029.ckpt')
resnet50 = model.net
# print(model)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        boxes, _ = mtcnn.detect(frame)
        faces = mtcnn(frame)
        if boxes is not None:
            face_box = []
            for box in boxes:
                bbox = list(map(int,box.tolist()))
                face_box.append(bbox)
                frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)

            for j, face in enumerate(faces):
                face = face.permute(1, 2, 0).numpy()*255
                h = face_box[j][3] - face_box[j][1]
                w = face_box[j][2] - face_box[j][0]
                
                landmarks = resnet50(transform(image = face)["image"].unsqueeze(0))[0]
                landmarks = (landmarks + 0.5) * torch.Tensor([w, h])
                x = torch.tensor([face_box[j][0],face_box[j][1]])
                landmarks = torch.add(landmarks, x)
                frame = paste_to_img(frame_bgr=frame, top_left=landmarks[17], top_right=landmarks[26])
                for i in range (landmarks.shape[0]):
                    frame = cv2.circle(frame, (int(landmarks[i, 0] ),int(landmarks[i, 1] )), radius=1, color=(255, 255, 0), thickness= 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    cv2.imshow('Face Detection', frame)
    
cap.release()
cv2.destroyAllWindows()
for i in range(30):
    cv2.waitKey(1)