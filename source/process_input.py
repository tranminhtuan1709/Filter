import numpy
import torch
import albumentations
import torch.optim
import mediapipe

import create_model


def get_bounding_box(image: numpy.ndarray) -> tuple:
    '''
        Get the largest bounding box around a human face in an image.

        Args:
            image (numpy.ndarray)
        
        Returns:
            A tuple contains coordinates of the top left point and the width
            and height of the largest bounding box.
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
    
    largest_bbox = bounding_boxes[0]

    for bbox in bounding_boxes:
        if bbox[2] * bbox[3] > largest_bbox[2] * largest_bbox[3]:
            largest_bbox = bbox
    
    return largest_bbox


def crop_face(bounding_box: tuple, image: numpy.ndarray) -> numpy.ndarray:
    '''
        Crop the human face following a given bounding box.

        Args:
            bounding_boxes (tuple)
            image (numpy.ndarray)

        Returns:
            The cropped face.
    '''

    x, y, w, h = bounding_box

    return image[y:y + h, x:x + w]


def get_landmarks(
    model: create_model.LandmarkDetectionModel,
    transformation: albumentations.Compose,
    image: numpy.ndarray,
    device: torch.device,
    bounding_box: tuple
) -> numpy.ndarray:
    '''
        Find landmark points in an image that containing only one object.

        Args:
            model (create_model.LandmarkDetectionModel)
            transformation (albumentations.Compose)
            faces (list)
            device (torch.device)
            bounding_box (tuple)

        Returns:
            68 landmark points for the object in the given image.
    '''

    image = transformation(image=image)['image'].unsqueeze(0)
    image = image.to(device)
    model.to(device)
    
    transformed_landmarks = model(image).squeeze(0).squeeze(0)
    transformed_landmarks = transformed_landmarks.cpu().detach().numpy()

    detransformed_landmarks = []

    x, y, w, h = bounding_box

    for point in transformed_landmarks:
        detransformed_landmarks.append([
                (point[0] + 0.5) * w + x,
                (point[1] + 0.5) * h + y
            ]
        )
    
    return numpy.int32(detransformed_landmarks)
