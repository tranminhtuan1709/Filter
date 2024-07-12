import matplotlib.patches
import matplotlib.pyplot
import swap_faces
import create_model
import torch
import albumentations
from albumentations.pytorch import ToTensorV2
import cv2
import process_input


model = create_model.LandmarkDetectionModel(68)
model.load_state_dict(torch.load('resources/trained_model.pth'))
device = torch.device('cpu')

transform_input = albumentations.Compose(
    [
        albumentations.Resize(height=224, width=224),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        
        ToTensorV2(),
    ],
    
    keypoint_params=albumentations.KeypointParams(
        format='xy',
        remove_invisible=False
    )
)

face_image_1 = cv2.imread('resources/my_face.jpg')
bounding_box_1 = process_input.get_bounding_box(face_image_1)
face_landmarks_1 = process_input.get_landmarks(
    model, transform_input, face_image_1, device, bounding_box_1
)

cap = cv2.VideoCapture(0)

while True:
    try:
        _, frame = cap.read()

        face_image_2 = frame
        bounding_box_2 = process_input.get_bounding_box(face_image_2)
        face_landmarks_2 = process_input.get_landmarks(
            model, transform_input, face_image_2, device, bounding_box_2
        )

        result = swap_faces.swap_faces(
            face_image_1, face_image_2, face_landmarks_1, face_landmarks_2
        )

        cv2.imshow('Result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        cv2.imshow('Result', face_image_2)
