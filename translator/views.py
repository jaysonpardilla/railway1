import numpy as np
import cv2
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
import tensorflow as tf
import mediapipe as mp
from django.conf import settings
import os

# model = tf.keras.models.load_model(os.path.join(settings.BASE_DIR, 'model.h5'))
model = tf.keras.models.load_model(settings.MODEL_PATH)
actions = ['hello', 'ikinagagalak kong makilala ka', 'magkita tayo bukas']  

mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    return pose.flatten()

def process_video(file_path):
    sequence = []
    cap = cv2.VideoCapture(file_path)
    step = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 30)

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        frame_count = 0
        while cap.isOpened() and len(sequence) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
            frame_count += 1

    while len(sequence) < 30:
        sequence.append(np.zeros(99))

    return np.array([sequence])  # shape (1, 30, 99)

class PredictView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        video = request.FILES.get('video')
        if not video:
            return Response({'error': 'No video uploaded'}, status=400)

        file_path = os.path.join(settings.MEDIA_ROOT, video.name)
        with open(file_path, 'wb+') as f:
            for chunk in video.chunks():
                f.write(chunk)

        sequence = process_video(file_path)
        prediction = model.predict(sequence)
        predicted_class = actions[np.argmax(prediction)]

        os.remove(file_path)  # Clean up

        return Response({'prediction': predicted_class})
