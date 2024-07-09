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
import process_input

    
def get_triangle_list(landmark: numpy.ndarray) -> numpy.ndarray:
    
    '''
        Get Delaunay Triangles from the list of landmark points.
        
        Args:
            landmarks (numpy.ndarray)
        
        Returns:
            A numpy.ndarray contains Delaunay Triangles, each triangle
            is presented by its coordinates x and y.
    '''

    triangle_list = []

    landmark = landmark.astype(numpy.float32)
    face_convexhull = cv2.convexHull(landmark)
    face_rect = cv2.boundingRect(face_convexhull)
    subdiv = cv2.Subdiv2D(face_rect)
    
    for point in landmark:
        subdiv.insert(point)
    
    triangles = subdiv.getTriangleList()

    for triangle in triangles:
        p1 = [triangle[0], triangle[1]]
        p2 = [triangle[2], triangle[3]]
        p3 = [triangle[4], triangle[5]]

        triangle_list.append([p1, p2, p3])
    
    return numpy.array(triangle_list, dtype=numpy.int32)

#_____________________________________________________________________________

def get_corresponding_triangles(
    triangle_list: numpy.ndarray,
    landmarks_1: numpy.ndarray,
    landmarks_2: numpy.ndarray
) -> numpy.ndarray:
    
    '''
        Get the list of Delaunay Triangles corresponding in index 
        with the given Delaunay Triangle list.
        
        Args:
            triangle_list (numpy.ndarray)
            landmarks_1 (numpy.ndarray)
            landmarks_2 (numpy.ndarray)
        
        Returns:
            A list containing numpy.ndarray, each numpy.ndarray
            contains coordinates of 3 points in landmarks_2 forming
            a triangle that corresponding in index with a triangle
            in the give triangle list.
    '''
    
    corresponding_triangles = []
    
    for triangle in triangle_list:
        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]
        
        index_p1 = numpy.where((landmarks_1 == p1).all(axis=1))[0][0]
        index_p2 = numpy.where((landmarks_1 == p2).all(axis=1))[0][0]
        index_p3 = numpy.where((landmarks_1 == p3).all(axis=1))[0][0]
                    
        corresponding_triangles.append([
                landmarks_2[index_p1],
                landmarks_2[index_p2],
                landmarks_2[index_p3]
            ]
        )
    
    return numpy.array(corresponding_triangles, dtype=numpy.int32)

#_____________________________________________________________________________


def get_rectangle_list(triangle_list: numpy.ndarray) -> numpy.ndarray:
    
    '''
    
    '''
    
    rectangle_list = []
    
    for triangle in triangle_list:
        rectangle_list.append(cv2.boundingRect(triangle))
    
    return numpy.array(rectangle_list, dtype=numpy.int32)


#_____________________________________________________________________________

def crop_triangle(
    triangle_list: numpy.ndarray,
    rectangle_list: numpy.ndarray,
    image: numpy.ndarray
) -> list:
    
    '''
    
    '''
    
    cropped_triangles = []
    
    for i in range(len(triangle_list)):
        p1 = triangle_list[i][0]
        p2 = triangle_list[i][1]
        p3 = triangle_list[i][2]
        
        rect_x, rect_y, rect_w, rect_h = rectangle_list[i]
        
        points = numpy.array(
            [[p1[0] - rect_x, p1[1] - rect_y],
             [p2[0] - rect_x, p2[1] - rect_y],
             [p3[0] - rect_x, p3[1] - rect_y]]
        )
        
        mask = numpy.zeros((rect_h, rect_w, 3), dtype=numpy.int32)
        cropped_frag = image[
            rect_y:rect_y + rect_h,
            rect_x:rect_x + rect_w
        ]
        
        cv2.fillConvexPoly(mask, points, (255, 255, 255))
        
        cropped_triangles.append(cv2.bitwise_and(
                cropped_frag,
                cropped_frag,
                mask
            )
        )

    return cropped_triangles



img = cv2.imread('resources/two_faces.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

transform_input = albumentations.Compose(
    [
        albumentations.Resize(height=224, width=224),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        ToTensorV2(),
    ]
)

model = create_model.LandmarkDetectionModel(68)
model.load_state_dict(torch.load('resources/trained_model.pth'))
device = torch.device('cpu')
model.to(device)

bbox = process_input.get_bounding_boxes(img)
faces = process_input.crop_faces(bbox, img)
landmarks = process_input.get_landmarks(model, transform_input, faces, device)
adjusted_landmarks = process_input.adjust_landmarks(landmarks, bbox)

face_1_triangles = get_triangle_list(adjusted_landmarks[0])
face_2_triangles = get_corresponding_triangles(
    face_1_triangles,
    adjusted_landmarks[0],
    adjusted_landmarks[1]
)

face_1_rectangles = get_rectangle_list(face_1_triangles)
face_2_rectangles = get_rectangle_list(face_2_triangles)

face_1_cropped_triangles = crop_triangle(
    face_1_triangles,
    face_1_rectangles,
    img
)

face_2_cropped_triangles = crop_triangle(
    face_2_triangles,
    face_2_rectangles,
    img
)


cv2.imshow('Final landmarks', img)

cv2.waitKey(0)
cv2.destroyAllWindows()