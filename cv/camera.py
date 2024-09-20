import cv2

from cv.face_processor import process_face

# Detect faces using a pre-trained face feature recognising classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Exit if you can't open camera
if not cap.isOpened():
    print("Can't open camera")
    exit()


# Continuously load frames from the camera
while True:
    ret, frame = cap.read()

    # Break out if you can't load frames
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale for better Haar Cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw a blue rect (it is in BGR) of thickness 2
        prediction = process_face(frame, (x, y, w, h))
        cv2.putText(
            frame,
            "Me" if prediction == 0 else "Not me",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if prediction == 0 else (0, 0, 255),
            2,
        )

    cv2.imshow("Camera", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 3: Release the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
