import xml.etree.ElementTree
import numpy
import torch
import matplotlib
import albumentations
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.models
import tqdm
import torch.optim
import cv2
import mediapipe
import pandas

import create_model


#_____________________________________________________________________________
def get_bounding_boxes(image: numpy.ndarray) -> list:
    
    '''
    
    '''
    
    bounding_boxes = []
    
    mp_face_detection = mediapipe.solutions.face_detection
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5)\
    as face_detection:
        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)
                
                bounding_boxes.append((x, y, w, h))
    
    return bounding_boxes

#_____________________________________________________________________________
def crop_faces(
    bounding_boxes: list,
    image: numpy.ndarray
) -> list:
    
    '''
    
    '''
    
    faces = []
    
    for x, y, w, h in bounding_boxes:
        faces.append(image[y:y + h, x:x + w])
    
    return faces

#_____________________________________________________________________________
def get_landmarks(
    model: create_model.LandmarkDetectionModel,
    transformation: albumentations.Compose,
    faces: list,
    device: torch.device
) -> numpy.ndarray:
    
    '''
    
    '''
    
    landmarks = []
    
    for face in faces:
        face = transformation(image=face)['image'].unsqueeze(0)
        face = face.to(device)
        model.to(device)
        
        landmark = model(face).squeeze(0).squeeze(0)
        
        landmarks.append(landmark.cpu().detach().numpy())
    
    return numpy.array(landmarks)

#_____________________________________________________________________________
def adjust_landmarks(
    landmarks: numpy.ndarray,
    bounding_boxes: list
) -> numpy.ndarray:
    
    '''
    
    '''
    
    adjusted_landmarks = []
    
    for i in range(len(bounding_boxes)):
        box_x, box_y, box_w, box_h = bounding_boxes[i]
        adjusted = []
        for x, y in landmarks[i]:
            x = (x + 0.5) * box_w + box_x
            y = (y + 0.5) * box_h + box_y

            adjusted.append([x, y])
        
        adjusted_landmarks.append(adjusted)
    
    return numpy.array(adjusted_landmarks, dtype=numpy.int32)
