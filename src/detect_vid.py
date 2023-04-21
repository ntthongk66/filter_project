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

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

#check device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

device = torch.device('cpu')
print(device)

#call an object from class 
mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

def create_bbox_cam():
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)
    while(True):
        
        isSuccess, frame = vid.read()
        if isSuccess:
            boxes, _ = mtcnn.detect(frame) # <class 'numpy.ndarray'>
            print(type(frame))
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[2], bbox[3]), (0,0,255), 2 )
                    # print((bbox[0], bbox[1]),(bbox[2], bbox[3]))
                    
                    
                    
                    
        cv2.imshow('Face_detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    vid.release()
    cv2.destroyAllWindows()
    
def create_bbox_vid():
    # detector = MTCNN()
    video = cv2.VideoCapture("E:/download/y2mate.com - Respect  WA_480p.mp4")

    # We need to check if camera
    # is opened previously or not
    if (video.isOpened() == False):
        print("Error reading video file")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    size = (frame_width, frame_height)

    # Below VideoWriter object will create
    # a frame of above defined The output
    # is stored in 'filename.avi' file.
    result = cv2.VideoWriter('E:/download/y2mate.com - Respect  WA_480p.avi',cv2.VideoWriter_fourcc(*'MJPG'),29, size)
    frame_num=0
    while (True):
        ret, frame = video.read()
        frame_num += 1
        print(frame_num)
        if ret == True:

            
            boxes, _ = mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    bbox = list(map(int, box.tolist()))
                    frame = cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[2], bbox[3]), (0,0,255), 2 )
            result.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break


    video.release()
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    
def create_bbox_img():
    path = "src\WIN_20200820_23_49_56_Pro.jpg"
    # img=cv2.imread(path)
    img_ = Image.open(path).convert('RGB')
    # im = np.array(img_)
    # print(im.shape)
    boxes, _ = mtcnn.detect(img_)
    if boxes is not None:
        for box in boxes:
            bbox = list(map(int, box.tolist()))
            img = ImageDraw.Draw(img_)
            # img.rectangle([(bbox[0], bbox[1]),(bbox[2], bbox[3])],outline="red")
            croped_img = img_.crop((bbox[0], bbox[1],bbox[2], bbox[3]))
            transform_resize = transforms.Resize((224, 224))
            # print(croped_img.shape)
            trans_croped_img = transform_resize(croped_img)
            print(croped_img.size)
            trans_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
            transform_to_tensor = transforms.ToTensor()
            trans_croped_img = transform_to_tensor(trans_croped_img)
            trans_croped_img = trans_normalize(trans_croped_img)
            
            trans_croped_img = trans_croped_img.unsqueeze(0)
            return trans_croped_img, img_, croped_img, bbox[0], bbox[1]
            print(trans_croped_img.shape)
                
            # img_ = cv2.rectangle(img, (bbox[0], bbox[1]),(bbox[2], bbox[3]), (0,0,255), 2 )
            

    # croped_img.save('src\h0.jpg')
    
    
def annotate_image(image: Image, landmark: np.ndarray) ->Image:
    draw = ImageDraw.Draw(image)
    for i in range(landmark.shape[0]):
        draw.ellipse((landmark[i][0] - 2, landmark[i][1] -2,
                    landmark[i][0] + 2, landmark[i][1] +2), fill = (255, 0, 0))
    return image
    
    
if __name__=="__main__":
    @hydra.main(version_base="1.3", config_path="../configs/model", config_name="dlib_resnet.yaml")
    def import_model(cfg: DictConfig) -> 'SimpleResnet':
        
        
        
        from src.models.dlib300w_module import DlibLiModule
        # print(type(dlib_module))
        # dlib_module.load_from_checkpoint('E:/filter_project/filter_project/logs/train/runs/2023-04-08_00-04-27/checkpoints/epoch_029.ckpt')
        m = DlibLiModule.load_from_checkpoint('E:/filter_project/filter_project/logs/train/runs/2023-04-08_00-04-27/checkpoints/epoch_029.ckpt').net
        input, ori_img, croped_img, left, top = create_bbox_img()
        
        output = m(input) #[1, 68, 2]
        h,w,_ = np.array(croped_img).shape
        output = torch.squeeze(output)
        # output = output.numpy()
        output = output.detach().numpy()
        
        # from src.data.components.dlib300W_Custom_dataset import TransformDataset
        # img = TransformDataset.annotate_tensor(input, output)
        
        # torchvision.utils.save_image(img, 'src/batch.jpg')
        # print(output)
        output = (output + 0.5)*np.array([w,h])
        output += np.array([left, top])
        output = output.astype('float32')
        
        
        print(type(output[0][1]))
        annotated_img = annotate_image(ori_img, output)
        annotated_img.save('src/a.jpg')
        
        
        
        
        # vid = cv2.VideoCapture(0)
        # vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)
        # while(True):
            
        #     isSuccess, frame = vid.read()
        #     if isSuccess:
        #         boxes, _ = mtcnn.detect(frame)
        #         if boxes is not None:
        #             for box in boxes:
        #                 bbox = list(map(int, box.tolist()))
        #                 frame = cv2.rectangle(frame, (bbox[0], bbox[1]),(bbox[2], bbox[3]), (0,0,255), 2 )
        #                 # print((bbox[0], bbox[1]),(bbox[2], bbox[3]))
        #                 print()
                        
        #     cv2.imshow('Face_detection', frame)
            
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
            
        # vid.release()
        # cv2.destroyAllWindows()
        
        
        
        
        
        
        
        # print(type(m))
        # img = Image.open('src\WIN_20200820_23_49_56_Pro.jpg')
        # crop_img_list = []
        # boxes, _ = mtcnn.detect(img)
        # if boxes is not None:
        #     for box in boxes:
        #         bbox = list(map(int, box.tolist()))
        #         crop_img_bbox = img.crop((bbox[0], bbox[1]),(bbox[2], bbox[3]))
        #         crop_img_list.append(crop_img_bbox)
        #         print(crop_img_bbox.shape)
        # output = m()
        # print(output.shape)
        # resnet50 = hydra.utils.instantiate(cfg.net)
        # print(type(resnet50))
        # model_weights = torch.load('E:/filter_project/filter_project/logs/train/runs/2023-04-08_00-04-27/checkpoints/epoch_020.ckpt')
        
        # m.load_state_dict(model_weights['state_dict'])
        return hydra.utils.instantiate(cfg.net)
    # create_bbox_cam()
    # create_bbox_vid()
    # create_bbox_img()
    import_model()
    # print(type(m))
    # print(type(torch.randn(1, 3, 224, 224)))
    # print(output.shape)
    # a = 1.
    # print(type(a))
