import cv2
import os
import sys

# Suppress system sounds related to OpenCV
sys.stderr = open(os.devnull, 'w')

# Your existing code continues below...
# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a directory to save the pictures if it doesn't exist
if not os.path.exists('captured_faces'):
    os.makedirs('captured_faces')

cap = cv2.VideoCapture(0) 
face_taken = False 

# Define the region where the face should be centered (e.g., the middle 30% of the frame)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

center_x_min = frame_width * 0.35
center_x_max = frame_width * 0.65
center_y_min = frame_height * 0.35
center_y_max = frame_height * 0.65

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Check if the face is centered in the defined region
        face_center_x = x + w / 2
        face_center_y = y + h / 2

        # Check if the face is in the center region (middle 30% of the frame)
        if (center_x_min < face_center_x < center_x_max) and (center_y_min < face_center_y < center_y_max):
            if not face_taken:
                # Crop the face from the frame
                face = frame[y:y + h, x:x + w]

                # Save the face image
                face_filename = f'captured_faces/face_{cv2.getTickCount()}.jpg'
                cv2.imwrite(face_filename, face)
                print(f"Face saved as {face_filename}")

                # Set the flag to True to prevent saving another face
                face_taken = True

                # Show the captured face in a new window (optional)
                cv2.imshow("Captured Face", face)
                cv2.waitKey(0) 

        else:
            face_taken = False 

    # Display the frame with the detected faces
    cv2.imshow("Webcam Feed with Face Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




