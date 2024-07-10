import numpy
import torch
import albumentations
import torch.optim
import mediapipe

import create_model


def get_bounding_boxes(image: numpy.ndarray) -> list:
    '''
        Get all bounding boxes around human faces in an image.

        Args:
            image (numpy.ndarray)
        
        Returns:
            A list containing all bounding boxes of human faces.
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
    
    return numpy.array(bounding_boxes, dtype=numpy.int32)


def crop_faces(bounding_boxes: list, image: numpy.ndarray) -> list:
    '''
        Crop human faces following the give bounding box list.

        Args:
            bounding_boxes (list)
            image (numpy.ndarray)

        Returns:
            A list containing parts of the given image that contain human faces.
    '''
    
    faces = []
    
    for x, y, w, h in bounding_boxes:
        faces.append(image[y:y + h, x:x + w])
    
    return faces


def get_landmark(
    model: create_model.LandmarkDetectionModel,
    transformation: albumentations.Compose,
    image: numpy.ndarray,
    device: torch.device
) -> numpy.ndarray:
    '''
        Find landmark points in an image that containing only one object.

        Args:
            model (create_model.LandmarkDetectionModel)
            transformation (albumentations.Compose)
            faces (list)
            device (torch.device)

        Returns:
            68 landmark points for the object in the given image.
    '''

    image = transformation(image=image)['image'].unsqueeze(0)
    face = image.to(device)
    model.to(device)
    
    landmark = model(image).squeeze(0).squeeze(0)
        
    return landmark.cpu().detach().numpy()


def adjust_landmark(
    landmark: numpy.ndarray,
    bounding_box: list
) -> numpy.ndarray:
    '''
        Adjust coordinates of landmark points following the size of the
        original image.

        Args:
            landmark (numpy.ndarray)
            bounding_box (list)

        Returns:
            Adjusted landmark points.
    '''
    
    adjusted_landmark = []
    print('bounding box haha:', bounding_box)

    box_x, box_y, box_w, box_h = bounding_box
    
    for x, y in landmark:
        x = (x + 0.5) * box_w + box_x
        y = (y + 0.5) * box_h + box_y
        
        adjusted_landmark.append([x, y])
    
    return numpy.array(adjusted_landmark, dtype=numpy.int32)
