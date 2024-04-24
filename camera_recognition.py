import cv2
from deepface import DeepFace

# Load pre-trained models for face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from webcam
    ret, frame = cap.read()
    
    # Run DeepFace emotion analysis on frame
    try:
        emotion = DeepFace.analyze(frame, actions='emotion')
        dominantEmotion = emotion[0]['dominant_emotion']
        # print(dominantEmotion)
    except:
        pass #skip if face not detected

    # Draw rectangle around the face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    # Add text to webcam video
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        cv2.putText(frame, dominantEmotion, (50,50), font, 3, (0,0,255), 2, cv2.LINE_4)
    except:
        pass #skip if face not detected

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
