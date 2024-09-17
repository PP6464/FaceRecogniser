# This is just to test loading the model and making a prediction with it
from PIL import Image
from keras.src.saving.saving_lib import load_model
import numpy as np

model = load_model("model/model.keras")
test_image = Image.open("data/me/me11.png").convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
test_image = np.array(test_image)
print(model.predict(test_image[None, :, :]))
