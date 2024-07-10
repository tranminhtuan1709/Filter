
import torch
import albumentations
from albumentations.pytorch import ToTensorV2
import torch.optim
import cv2

import delaunay_triangle
import process_input
import create_model
import apply_filter


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read the frame.")
            break

        img = frame

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

        bboxes = process_input.get_bounding_boxes(img)
        faces = process_input.crop_faces(bboxes, img)
        landmarks = []
        for face in faces:
            landmarks.append(process_input.get_landmark(
                    model, transform_input, face, device
                )
            )

        adjusted_landmarks = []
        for i in range(len(landmarks)):
            adjusted_landmarks.append(process_input.adjust_landmark(
                    landmarks[i], bboxes[i]
                )
            )
        
        filter_image, filter_landmark = apply_filter.load_filter('resources/filters/anonymous')

        faces2 = filter_image
        landmarks2 = process_input.get_landmark(model, transform_input, faces2, device)
        adjusted_landmarks2 = process_input.adjust_landmark(landmarks2, (0, 0, filter_image.shape[1], filter_image.shape[0]))

        filter_landmark = adjusted_landmarks2[0]

        result = apply_filter.apply_filter(
            face_image=img,
            face_landmark=adjusted_landmarks[0],
            filter_image=filter_image,
            filter_landmark=filter_landmark
        )

        cv2.imshow('Result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
