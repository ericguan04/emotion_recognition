import cv2
from deepface import DeepFace

img = cv2.imread('happy_test.jpg')
emotion = DeepFace.analyze(img, actions='emotion')
dominantEmotion = emotion[0]['dominant_emotion']
print(dominantEmotion)