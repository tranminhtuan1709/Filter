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
from skimage.exposure import match_histograms

import delaunay_triangle
import process_input
import create_model

#_____________________________________________________________________________
def load_filter(filter_path: str) -> tuple:
    
    '''
    
    '''
    
    filter_image = cv2.imread(filter_path + '.png')
    filter_image = cv2.cvtColor(filter_image, cv2.COLOR_BGR2RGB)
    
    df = pandas.read_csv(filter_path + '_annotations.csv')
    
    x = df.iloc[:, 1].values
    y = df.iloc[:, 2].values
    
    filter_landmark = numpy.column_stack((x, y))

    return filter_image, filter_landmark

#_____________________________________________________________________________
def show_triangles(triangles, img):
    for triangle in triangles:
        p1 = triangle[0]
        p2 = triangle[1]
        p3 = triangle[2]

        cv2.line(img, p1, p2, color=(0, 255, 0), thickness=1)
        cv2.line(img, p1, p3, color=(0, 255, 0), thickness=1)
        cv2.line(img, p3, p2, color=(0, 255, 0), thickness=1)
    
    cv2.imshow('ABC', img)

#_____________________________________________________________________________

def show_landmark(landmarks, img):
    for landmark in landmarks:
        cv2.circle(img, landmark, radius=2, color=(255, 0, 0), thickness=1)
    
    cv2.imshow('ABC', img)
#_____________________________________________________________________________
def apply_filter(
    face_image: numpy.ndarray,
    face_landmark: numpy.ndarray,
    filter_image: numpy.ndarray,
    filter_landamrk: numpy.ndarray,
) -> numpy.ndarray:
    
    '''
    
    '''
    
    face_triangles = delaunay_triangle.get_triangle_list(face_landmark)
    filter_triangles = delaunay_triangle.get_corresponding_triangles(
        triangle_list=face_triangles,
        landmarks_1=face_landmark,
        landmarks_2=filter_landamrk
    )

    face_rectangles = delaunay_triangle.get_rectangle_list(
        triangle_list=face_triangles
    )

    filter_rectangles = delaunay_triangle.get_rectangle_list(
        triangle_list=filter_triangles
    )
        
    filter_cropped_triangles = delaunay_triangle.crop_triangle(
        triangle_list=filter_triangles,
        rectangle_list=filter_rectangles,
        image=filter_image
    )

    filter_affine_triangles = []

    cv2.fillConvexPoly(
        img=face_image,
        points=cv2.convexHull(points=face_landmark),
        color=(0, 0, 0)
    )

    for i in range(len(face_triangles)):
        triangle_1 = numpy.float32(face_triangles[i])
        triangle_2 = numpy.float32(filter_triangles[i])
        
        p1 = triangle_1[0]
        p2 = triangle_1[1]
        p3 = triangle_1[2]

        p4 = triangle_2[0]
        p5 = triangle_2[1]
        p6 = triangle_2[2]

        x1, y1, w1, h1 = face_rectangles[i]
        x2, y2, w2, h2 = filter_rectangles[i]

        points_1 = numpy.array(
            [[p1[0] - x1, p1[1] - y1],
             [p2[0] - x1, p2[1] - y1],
             [p3[0] - x1, p3[1] - y1]], dtype=numpy.float32
        )

        points_2 = numpy.array(
            [[p4[0] - x2, p4[1] - y2],
             [p5[0] - x2, p5[1] - y2],
             [p6[0] - x2, p6[1] - y2]], dtype=numpy.float32
        )

        M = cv2.getAffineTransform(points_2, points_1)

        warped_triangle = cv2.warpAffine(
            src=filter_cropped_triangles[i],
            M=M,
            dsize=(w1, h1)
        )

        filter_affine_triangles.append(warped_triangle)

    for i in range(len(face_triangles)):
        x, y, w, h = face_rectangles[i]
        dst_area = face_image[y:y + h, x:x + w]
        dst_area = cv2.add(
            src1=filter_affine_triangles[i],
            src2=dst_area,
        )

        face_image[y:y + h, x:x + w] = dst_area
    
    return face_image
    


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

filter_image, filter_landmark = load_filter('resources/filters/anonymous')

result = apply_filter(
    face_image=img,
    face_landmark=adjusted_landmarks[0],
    filter_image=img,
    filter_landamrk=adjusted_landmarks[1]
)

cv2.imshow('Result', result)


cv2.waitKey(0)
cv2.destroyAllWindows()