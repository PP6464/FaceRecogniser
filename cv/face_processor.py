import cv2
import numpy as np
from keras.src.saving import load_model


def process_face(image, face_coords) -> int:
    x, y, w, h = face_coords
    face = image[y:y+h, x:x+w]

    face_resized = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (224, 224))

    face_array = np.array(face_resized)

    model = load_model("../nn/model/model.keras")
    raw_prediction = model.predict(face_array[None, :, :])

    P_me = raw_prediction[0][0]  # Probability it is me
    P_not_me = raw_prediction[0][1]  # Probability it is not me

    if P_not_me >= P_me:
        return 1  # Same as the label for 'not me'
    else:
        return 0  # Same as the label for 'me'
