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

def get_landmarks(
    model: LandmarkDetectionModel,
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
        
        landmark = model(face)
        
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
        x = bounding_boxes[i][0]
        y = bounding_boxes[i][1]
        
        landmarks[i] += (x, y)
        
        adjusted_landmarks.append(landmarks[i])
    
    return adjusted_landmarks