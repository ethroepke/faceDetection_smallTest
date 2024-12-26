import cv2
import os
import sys
import numpy as np
import pyttsx3  

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set properties for speech (speed rate and volume)
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

# Load Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load saved faces
saved_faces = []

# Load all images in the 'captured_faces' directory
captured_faces_path = 'captured_faces'
for filename in os.listdir(captured_faces_path):
    if filename.endswith(".jpg"):
        face = cv2.imread(os.path.join(captured_faces_path, filename))
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        saved_faces.append(face_gray)

# Function to match faces using normalized correlation
def match_face(face, saved_face):
    # Resize detected face to match saved face dimensions
    face_resized = cv2.resize(face, (saved_face.shape[1], saved_face.shape[0]))
    result = cv2.matchTemplate(face_resized, saved_face, cv2.TM_CCOEFF_NORMED)
    return np.max(result)

# Function to check if any saved face matches
def check_for_face_match(face, threshold=0.7):
    for saved_face in saved_faces:
        match_score = match_face(face, saved_face)
        if match_score > threshold:
            return True
    return False

# Create a directory to save the pictures if it doesn't exist
if not os.path.exists('captured_faces'):
    os.makedirs('captured_faces')

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)


# State to track if the last detection was a match
last_match_state = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Initialize a flag to check if there's a match
    match_found = False

    # Display detected faces in the frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Check if the detected face matches any saved face
        face_to_check = gray[y:y + h, x:x + w]
        if check_for_face_match(face_to_check):
            match_found = True

            # Display "Face Match" above the bounding box
            text = "Face Match"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y - 10  # Position text above the box
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            # Display "No Face Match" above the bounding box for non-matching faces
            text = "No Face Match"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y - 10  # Position text above the box
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Announce "Face Identified" only once per match state transition
    if match_found and not last_match_state:
        engine.say("Face Identified")
        engine.runAndWait()

    # Update last match state
    last_match_state = match_found

    # Display the frame with the detected faces
    cv2.imshow("Webcam Feed with Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


