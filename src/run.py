import cv2
import imutils
import math
from facenet_pytorch import MTCNN
import torch
import numpy as np
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


######################################FIRST VER OF FILTER :)))##################################################
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
##########################################################################################


########################################SECOND VER OF FILTER##############################
import mediapipe as mp
import FaceMesh.faceBlendCommon as fbc
import csv


filters_config = {
    'anonymous':
        [{'path': "src/FaceMesh/filters/anonymous.png",
          'anno_path': "src/FaceMesh/filters/anonymous_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    # 'anime':
    #     [{'path': "filters/anime.png",
    #       'anno_path': "filters/anime_annotations.csv",
    #       'morph': True, 'animated': False, 'has_alpha': True}],
    # 'dog':
    #     [{'path': "filters/dog-ears.png",
    #       'anno_path': "filters/dog-ears_annotations.csv",
    #       'morph': False, 'animated': False, 'has_alpha': True},
    #      {'path': "filters/dog-nose.png",
    #       'anno_path': "filters/dog-nose_annotations.csv",
    #       'morph': False, 'animated': False, 'has_alpha': True}],
    # 'cat':
    #     [{'path': "filters/cat-ears.png",
    #       'anno_path': "filters/cat-ears_annotations.csv",
    #       'morph': False, 'animated': False, 'has_alpha': True},
    #      {'path': "filters/cat-nose.png",
    #       'anno_path': "filters/cat-nose_annotations.csv",
    #       'morph': False, 'animated': False, 'has_alpha': True}],
    # 'jason-joker':
    #     [{'path': "filters/jason-joker.png",
    #       'anno_path': "filters/jason-joker_annotations.csv",
    #       'morph': True, 'animated': False, 'has_alpha': True}],
    # 'gold-crown':
    #     [{'path': "filters/gold-crown.png",
    #       'anno_path': "filters/gold-crown_annotations.csv",
    #       'morph': False, 'animated': False, 'has_alpha': True}],
    # 'flower-crown':
    #     [{'path': "filters/flower-crown.png",
    #       'anno_path': "filters/flower-crown_annotations.csv",
    #       'morph': False, 'animated': False, 'has_alpha': True}],
}

def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((b, g, r))

    return img, alpha

def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points


def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter(filter_name="anonymous"):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime


##########################################################################################
model = DlibLiModule.load_from_checkpoint(checkpoint_path='E:/filter_project/filter_project/logs/train/runs/2023-04-08_00-04-27/checkpoints/epoch_029.ckpt')
resnet50 = model.net
# print(model)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,400)

visualize_kpt = False

count = 0
isFirstFrame = True
sigma = 50

iter_filter_keys = iter(filters_config.keys())
filters, multi_filter_runtime = load_filter(next(iter_filter_keys))

while cap.isOpened():
    isSuccess, frame = cap.read()
    if isSuccess:
        boxes, _ = mtcnn.detect(frame)
        faces = mtcnn(frame)
        
        img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
                landmarks = landmarks.detach().numpy()
                
                for idx, filter in enumerate(filters):

                    filter_runtime = multi_filter_runtime[idx]
                    img1 = filter_runtime['img']
                    points1 = filter_runtime['points']
                    img1_alpha = filter_runtime['img_a']

                    if filter['morph']:

                        hullIndex = filter_runtime['hullIndex']
                        dt = filter_runtime['dt']
                        hull1 = filter_runtime['hull']

                        # create copy of frame
                        warped_img = np.copy(frame)

                        # Find convex hull
                        hull2 = []
                        for i in range(0, len(hullIndex)):
                            hull2.append(landmarks[hullIndex[i][0]])

                        mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                        mask1 = cv2.merge((mask1, mask1, mask1))
                        img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                        # Warp the triangles
                        for i in range(0, len(dt)):
                            t1 = []
                            t2 = []

                            for j in range(0, 3):
                                t1.append(hull1[dt[i][j]])
                                t2.append(hull2[dt[i][j]])

                            fbc.warpTriangle(img1, warped_img, t1, t2)
                            fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                        # Blur the mask before blending
                        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                        mask2 = (255.0, 255.0, 255.0) - mask1

                        # Perform alpha blending of the two images
                        temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                        temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                        output = temp1 + temp2
                    else:
                        dst_points = [landmarks[int(list(points1.keys())[0])], landmarks[int(list(points1.keys())[1])]]
                        tform = fbc.similarityTransform(list(points1.values()), dst_points)
                        # Apply similarity transform to input image
                        trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                        trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                        mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                        # Blur the mask before blending
                        mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                        mask2 = (255.0, 255.0, 255.0) - mask1

                        # Perform alpha blending of the two images
                        temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                        temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                        output = temp1 + temp2

                    frame = output = np.uint8(output)
                
                print(type(landmarks))
                # print(landmarks.shape) #[68, 2]
                # frame = paste_to_img(frame_bgr=frame, top_left=landmarks[17], top_right=landmarks[26])
                if(visualize_kpt):
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