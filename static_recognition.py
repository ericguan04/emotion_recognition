'''
static_recognition.py returns facial emotion for an image passed in
'''

import cv2
from deepface import DeepFace
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# img = cv2.imread('happy_test.jpg')

# detect multiple faces
faces = RetinaFace.extract_faces('happy_test_two.jpg')
print("There are", len(faces), "faces in the image")

for face in faces:
    # plt.imshow(face)
    # plt.axis('off')
    # plt.show()
    emotion = DeepFace.analyze(face, detector_backend='skip', actions='emotion')
    dominantEmotion = emotion[0]['dominant_emotion']
    print(dominantEmotion)