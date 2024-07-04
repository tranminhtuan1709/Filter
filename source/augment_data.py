import albumentations
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2
import numpy
import torch

transform_train = albumentations.Compose([
    albumentations.Resize(height=224, width=224),
    albumentations.RGBShift(
        r_shift_limit=(-20, 20),
        g_shift_limit=(-20, 20),
        b_shift_limit=(-20, 20),
        p=1.0
    ),
    albumentations.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.01),
    albumentations.Rotate(limit=(-45, 45), p=1.00),
    albumentations.RandomBrightnessContrast(p=0.5),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], keypoint_params=albumentations.KeypointParams(format='xy', remove_invisible=False))

transform_test = albumentations.Compose([
    albumentations.Resize(height=224, width=224),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], keypoint_params=albumentations.KeypointParams(format='xy', remove_invisible=False))

def augment_data(cropped_images: list,
                       adjusted_landmarks: list,
                       transform_train: albumentations.Compose) -> tuple:
    '''
        Augment data.

        Args:
            cropped_images (list)
            adjusted_landmarks (list)
            transform_train (albumentation.Compose)
        
        Returns:
            augmented_images (list): a list containing all augmented images.
            augmented_landmarks (list): a list containing all augmented landmarks.
    '''

    augmented_images = []
    augmented_landmarks = []

    for i in range(len(cropped_images)):
        augment = transform_train(image=numpy.array(cropped_images[i]),
                                  keypoints=adjusted_landmarks[i])
        
        image = augment['image']
        channels, width, height = augment['image'].shape
        keypoint = augment['keypoints'] / numpy.array([width, height]) - 0.5
        keypoint = torch.tensor(data=keypoint, dtype=float)
        channels = channels

        augmented_images.append(image)
        augmented_landmarks.append(keypoint)
    
    return augmented_images, augmented_landmarks
