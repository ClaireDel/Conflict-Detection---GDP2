from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time
import tensorflow as tf
from track.tracking import *


import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform


a = time.perf_counter()
COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
print(CTX, torch.cuda.current_device(), torch.cuda.device_count(), torch.cuda.get_device_name(0))

def draw_pose(keypoints,img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS,2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0],keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0],keypoints[kpt_b][1] 
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

def draw_bbox(box,img):
    """draw the detected bounding box on the image.
    :param img:
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score)<threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def argmax_plus(d2_array, threshold=0.5, favored=1):
  end_list = []
  for row in d2_array:
    if row[favored]>threshold:
      end_list.append(favored)
    else:
      end_list.append(1-favored)
  return np.array(end_list)
# filename = ''

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='demo/inference-config.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam', default = 0, action='store_true') # default = 1
    parser.add_argument('--image', type=str, default='C:\\Users\\Marc-Olivier\\Downloads\\bis\kicking\\a05_s02_00139_color.png' ) # default='/Users/clair/Desktop/danse.jpg'default =filename
    parser.add_argument('--write', default = 1, action='store_true')
    parser.add_argument('--showFps',action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase  
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def flip_kpt(kpt_vector):
    end_vector = [kpt_vector[k]*(-1)**(k+1) for k in range(len(kpt_vector))]
    return end_vector

def special_normalisation(keypoint_vector):
    end_vector = [0]*len(keypoint_vector)
    new_origin = keypoint_vector[0], keypoint_vector[1]
    for k in range(0, len(keypoint_vector), 2):
        end_vector[k] = keypoint_vector[k] - new_origin[0]
        end_vector[k+1] = keypoint_vector[k+1] - new_origin[1]

    return end_vector
def spe_norm(d2_skeleton, box):
    size = box[1][1]-box[0][1], box[1][0]-box[0][0]
    origin = d2_skeleton[0]*1
    for k in range(len(d2_skeleton)):
        d2_skeleton[k] = d2_skeleton[k] - origin
        for j in range(len(d2_skeleton[0])):
            d2_skeleton[k][j] = d2_skeleton[k][j]/size[1-j]
    return d2_skeleton
def draw_label(num, image_bgr, box ,number_classes=2):
    if number_classes == 4:
        if num == 0:
            cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(255,0, 0),thickness=3)
            cv2.putText(image_bgr, f'{label_dict[num]}', (int(box[0][0]),int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        elif num==1:
            cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(255,255, 0),thickness=3)
            cv2.putText(image_bgr, f'{label_dict[num]}', (int(box[0][0]),int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        elif num ==2:
            cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(100,100, 0),thickness=3)
            cv2.putText(image_bgr, f'{label_dict[num]}', (int(box[0][0]),int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        del num
    elif number_classes == 2:
        if num == 0:
            cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(0,255, 0),thickness=3)
        elif num==1:
            cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(0,0, 255),thickness=3)
   
    


##-------------------- SETUP - MODEL LOADING ----------------------------
# for filename in os.listdir(folder_in):
#     PATH = folder_in + filename

# cudnn related setting
cudnn.benchmark = cfg.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
a= time.perf_counter_ns()
args = parse_args()
update_config(cfg, args)

box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
box_model.to(CTX)
box_model.eval()

pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
    cfg, is_train=False
)
with tf.device('/CPU:0'):
    pose_eval_model = tf.keras.models.load_model('.\\NNNN')


if cfg.TEST.MODEL_FILE:
    print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False) #map_location=torch.device('cpu')
else:
    print('expected model defined in config at TEST.MODEL_FILE')

pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
pose_model.to(CTX)
pose_model.eval()
print('time', (time.perf_counter_ns()-a)/10e9)

##--------------------END SETUP - MODEL LOADING ----------------------------

##-------------------- USE SELECTION ---------------------------

label_dict = {0:'punch', 1:'kick', 2:'push', 3:'neutral'}
def video_process(from_webcam=False, video_path='.', write=False, _save_path='demo.avi', show_fps=False):
# Loading an video or an image or webcam 

    persons = []
    if from_webcam:
        print("webcam")
        vidcap = cv2.VideoCapture(0)   
    else:
        vidcap = cv2.VideoCapture(video_path) 
    
    if write:
        save_path = _save_path
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_path,fourcc, vidcap.get(cv2.CAP_PROP_FPS), (int(vidcap.get(3)),int(vidcap.get(4)))) #24
    i=0
    while True:
        #print(i)
        i+=1
        ret, image_bgr = vidcap.read()
        #print(ret)
        if ret:
            #print("a")
            last_time = time.time()
            image = image_bgr[:, :, [2, 1, 0]]

            input = []
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
            input.append(img_tensor)

            # object detection box
            pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)
            
            # pose estimation
            if len(pred_boxes) >= 1:
                for box in pred_boxes:
                    
                    '''cv2.rectangle(image_bgr, 
                        (int(box[0][0]),int(box[0][1])), 
                        (int(box[1][0]),int(box[1][1])),
                        color=(0, 255, 0),
                        thickness=3)
                    '''                      
                    
                    center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                    """if len(persons) == 0:
                        person = Person(position=center, box=box)
                        persons.append(person)
                    else:
                        found_corresponding = False
                        for p in persons:
                            if p.is_you(box):
                                sp, ep = get_overlap_rectangle(p.bounding_box, box)
                                #print(sp, ep)
                                found_corresponding = True
                                p.update_position(center, box)
                                person = p
                                break
                        if not found_corresponding:
                            person = Person(position=center, box=box)
                            persons.append(person)"""


                    image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                    pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                    if len(pose_preds)>=1:
                        for kpt in pose_preds:
                            draw_pose(kpt,image_bgr) # draw the poses
                        
                            L = np.array(kpt)
                            size = box[1][1]-box[0][1], box[1][0]-box[0][0]
                            origin = L[0]*1
                            for k in range(len(L)):
                                L[k] = L[k] - origin
                                for j in range(len(L[0])):
                                    L[k][j] = L[k][j]/size[1-j]
                            a = np.reshape(L,(1,34))
                            a = a[0]
                           
                            a = special_normalisation(a)
                                
                            person.add_pose(L)
                                
                            with tf.device('/CPU:0'):
                                _num = pose_eval_model.predict(np.array([a]))
                                num = argmax_plus(_num, threshold=0.6, favored=1)[0]
                                draw_label(num, image_bgr, box, number_classes=2)



                                '''
                                cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(0,255, 0),thickness=3)
                                if num == 1:
                                    cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(150,150, 255),thickness=3)
                                    cv2.putText(image_bgr, f'{_num[0][num]:0.2f}', (int(box[0][0])+50,int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 60), 2)   
                                    if person.aggressive():
                                        cv2.putText(image_bgr, f'{_num[0][num]:0.2f}', (int(box[0][0])+50,int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 60), 2)
                                        cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(0,0, 255),thickness=3)
                            #calc = np.reshape(kpt, (1,34))
                            #num = pose_eval_model.predict(calc)                            
                            #img = cv2.putText(image_bgr, f'{num}', (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                                '''
            if show_fps:
                fps = 1/(time.time()-last_time)
                img = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

            if write:
                out.write(image_bgr)
            
            if from_webcam:
                cv2.imshow('demo',image_bgr)
                if cv2.waitKey(1) & 0XFF==ord('q'):
                    break
            #if cv2.waitKey(1) & 0XFF==ord('q'):
            #    break
        else:
            print('cannot load the video.')
            break

    cv2.destroyAllWindows()
    vidcap.release()
    if write:
        print('video has been saved as {}'.format(save_path))
        out.release()
            
            

def image_process(img_path, write=False, _save_path='demo.avi', show_fps=False):
    persons = []
    # estimate on the image
    image_bgr = cv2.imread(img_path)
    last_time = time.time()
    image = image_bgr[:, :, [2, 1, 0]]

    input = []
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
    input.append(img_tensor)

    # object detection box
    pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

    # pose estimation
    if len(pred_boxes) >= 1:
            for box in pred_boxes: 
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                if len(pose_preds)>=1:
                    for kpt in pose_preds:
                        draw_pose(kpt,image_bgr) # draw the poses
                        
                        #cv2.putText(image_bgr, f'{person.id}', (int(person.position[0]), int(person.position[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                        L = np.array(kpt)
                        #size = image_bgr.shape
                        size = box[1][1]-box[0][1], box[1][0]-box[0][0]
                        origin = L[0]*1
                        for k in range(len(L)):
                            L[k] = L[k] - origin
                            for j in range(len(L[0])):
                                L[k][j] = L[k][j]/size[1-j]
                        normalized_skeleton = np.reshape(L,(1,34))
                        normalized_skeleton = normalized_skeleton[0]
                            
                        with tf.device('/CPU:0'):
                            _num = pose_eval_model.predict(np.array([normalized_skeleton]))
                            num = argmax_plus(_num, threshold=0.6, favored=1)[0]
                            draw_label(num, image_bgr, box, number_classes=2)
                            '''if person.aggressive():
                                    cv2.putText(image_bgr, f'{_num[0][num]:0.2f}', (int(box[0][0])+50,int(box[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 60), 2)
                                    cv2.rectangle(image_bgr, (int(box[0][0]),int(box[0][1])), (int(box[1][0]),int(box[1][1])) , color=(0,0, 255),thickness=3)'''

    if show_fps:
        fps = 1/(time.time()-last_time)
        img = cv2.putText(image_bgr, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    
    if write:
        save_path = _save_path # folder_exit + filename 
        cv2.imwrite(save_path,image_bgr)
        print('the result image has been saved as {}'.format(save_path))

    # cv2.imshow('demo',image_bgr)
    if cv2.waitKey(0) & 0XFF==ord('q'):
        cv2.destroyAllWindows()


#video_process(from_webcam=False, video_path='data\\seq7.avi',write=True, show_fps=True, _save_path='demoNNNN3.avi')
video_process(video_path="data\\pc\\sc11_1.avi",from_webcam=False, write=True, _save_path='sc11_1_nnn.avi', show_fps=True)
