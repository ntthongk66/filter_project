import cv2
from facenet_pytorch import MTCNN
import torch
from src.models.components.simple_resnet import SimpleResnet
from src.models import dlib300w_module
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import hydra
import pyrootutils
import torchvision
from src.models.dlib300w_module import DlibLiModule

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
device = torch.device('cpu')
print(device)
#call an object from class 
mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

def face_cam():
    
    # defind model 
    resnet50 = DlibLiModule.load_from_checkpoint(checkpoint_path='E:/filter_project/filter_project/logs/train/runs/2023-04-08_00-04-27/checkpoints/epoch_029.ckpt').net
              
    transform = transforms.Compose([
                transforms.Resize(224, 224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                
            ])
    
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)
    
    
    while(True):
        
        isSuccess, frame = vid.read()
        if isSuccess:
            boxes, _ = mtcnn.detect(frame) # <class 'numpy.ndarray'>
            # print(type(frame))
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[2], bbox[3]), (0,0,255), 2 )
                    # print((bbox[0], bbox[1]),(bbox[2], bbox[3]))\
                    
                    
                    
                    
                    
        cv2.imshow('Face_detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    vid.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    # face_cam()
    img = cv2.imread('src/h0.jpg')
    print(type(img))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    print('img_pil', type(im_pil))
    # For reversing the operation:
    im_np = np.asarray(im_pil)
    print('img_np', type(im_np))
    img_ = Image.open('src/h0.jpg')
    print('PIL: ',type(img_))