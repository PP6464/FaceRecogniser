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
        # Normally the Haar Cascade detects just the face, not the surrounding hair etc.
        # The model has been trained with the face, hair and backgrounds in the data
        # To compensate for this, we will need to enlarge the frame in both directions about the centre
        new_w = int(1.15 * w)
        new_h = int(1.3 * h)
        new_x = int((x + w / 2) - new_w / 2)
        new_y = int((y + h / 2) - new_h / 2)
        if new_x < 0:
            new_x = 0
        if new_y < 0:
            new_y = 0
        if new_x > 1919:
            new_x = 1919
        if new_y > 1079:
            new_y = 1079
        # Draw a blue rect (it is in BGR) of thickness 2
        cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (255, 0, 0), 2)
        prediction = process_face(gray, (new_x, new_y, new_w, new_h))
        cv2.putText(
            frame,
            "Me" if prediction == 0 else "Not me",
            (new_x, new_y - 10),
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
