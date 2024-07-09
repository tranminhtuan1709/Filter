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
class Filter():
    def __init__(self) -> None:
        
        '''
            Initianize a Filter object.
            
            Args:
            
            Returns:
                None
        '''
        
        self.filter_image = None
        self.filter_landmark = None
    
    def load_filter(self, filter_path: str) -> None:
        
        '''
        
        '''
        
        self.filter_image = cv2.imread(filter_path + '.png')
        self.filter_image = cv2.cvtColor(
            self.filter_image, cv2.COLOR_BGR2RGB
        )
        
        df = pandas.read_csv(filter_path + '_annotations.csv')
        
        x = df.iloc[:, 1].values
        y = df.iloc[:, 2].values
        
        self.filter_landmark = numpy.column_stack((x, y))
    
    def apply_filter(
        self,
        image: numpy.ndarray,
        image_landmark: numpy.ndarray,
        dtri: DelaunayTriangle
    ) -> numpy.ndarray:
        
        '''
        
        '''
        
        filter_triangles = dtri.get_triangle_list(self.filter_landmark)
        image_triangles = dtri.get_corresponding_triangles(
            filter_triangles,
            self.filter_landmark,
            image_landmark
        )
        
        filter_rectangles = dtri.get_rectangle_list(filter_triangles)
        image_rectangles = dtri.get_rectangle_list(image_triangles)
        
        cropped_filter_triangles = dtri.crop_triangle(
            filter_triangles,
            filter_rectangles,
            self.filter_image
        )
        
        new_image = numpy.full(image.shape, 255)
        
        cv2.fillConvexPoly(
            image,
            cv2.convexHull(image_landmark),
            (255, 255, 255)
        )
        
        cv2.fillConvexPoly(
            new_image,
            cv2.convexHull(image_landmark),
            (0, 0, 0)
        )
        
        for i in range(len(filter_triangles)):
            triangle_1 = numpy.float32(filter_triangles[i])
            triangle_2 = numpy.float32(image_triangles[i])
            
            x, y, w, h = image_rectangles[i]
            
            M = cv2.getAffineTransform(triangle_1, triangle_2)
            
            warped_triangle = cv2.warpAffine(
                cropped_filter_triangles[i],
                M,
                (w, h)
            )
            
            dest_area = new_image[y:y + h, x:x + w]
            dest_area = cv2.add(dest_area, warped_triangle)
            new_image[y:y + h, x:x + w] = dest_area
        
        new_image = cv2.bitwise_and(new_image, new_image, image)
        
        return new_image