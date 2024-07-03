import xml
import xml.etree
import xml.etree.ElementTree
import numpy
from PIL import Image

def create_an_image_dict(image : xml.etree.ElementTree) -> dict:
    '''
        Create an image dictionary containing image details.

        Args:
            image (xml.etree.ElementTree) : an XML element containing an image.
        Returns:
            A dictionary containing information of the image:
            - filename (str)
            - image_width (int)
            - image_height (int)
            - box_top (int)
            - box_left (int)
            - box_width (int)
            - box_height (int)
            - landmark_list (numpy.array)
    '''

    image_dict = {}
    image_dict['filename'] = image.attrib['file']
    image_dict['image_width'] = int(image.attrib['width'])
    image_dict['image_height'] = int(image.attrib['height'])

    box = image.find('box')

    image_dict['box_top'] = int(box.attrib['top'])
    image_dict['box_left'] = int(box.attrib['left'])
    image_dict['box_width'] = int(box.attrib['width'])
    image_dict['box_height'] = int(box.attrib['height'])

    landmark_tags = box.findall('part')
    landmark_list = numpy.array([(int(landmark_tag.attrib['x']),
                                  int(landmark_tag.attrib['y']))
                                  for landmark_tag in landmark_tags])
    
    image_dict['landmark_list'] = landmark_list

    return image_dict

def create_image_list(root: xml.etree.ElementTree) -> list:
    '''
        Create a list containing dictionaries of all images.

        Args:
            root (xml.etree.ElementTree)

        Returns:
            A list containing dictionaries of all images.
    '''

    images = root.find('images')
    image_list = []

    for child in images:
        image_list.append(create_an_image_dict(child))
    
    return image_list

def crop_images(image_list: list) -> tuple:
    '''
        Crop all original images and adjust landmark points.

        Agrs:
            image_list (list): a list contiaining dictionaries of all images.

        Returns:
            cropped_images, adjusted_landmarks (tuple): a tuple containing two lists, the
            first list contains all cropped images in type PIL.Image, the second list contains
            landmark points of each image in type numpy array.
    '''
    
    cropped_images = []
    adjusted_landmarks = []

    for image_dict in image_list:
        image = Image.open('resources\\' + image_dict['filename'])
        
        box_top = image_dict['box_top']
        box_left = image_dict['box_left']
        box_bottom = image_dict['box_top'] + image_dict['box_height']
        box_right = image_dict['box_left'] + image_dict['box_width']

        image = image.crop((box_left, box_top, box_right, box_bottom))

        landmarks = image_dict['landmark_list'] - numpy.array([(box_left, box_top)])

        cropped_images.append(image)
        adjusted_landmarks.append(landmarks)

    return cropped_images, adjusted_landmarks
