# This is just to test loading the model and making a prediction with it
from PIL import Image
from keras.src.saving.saving_lib import load_model
import numpy as np

model = load_model("model/model.keras")

test_images = []
for i in range(60):
    img_me = Image.open(f"data/me/me{i+1}.png").convert('L').resize((512, 512), Image.Resampling.LANCZOS)
    img_not_me = Image.open(f"data/notme/notme{i+1}.png").convert('L').resize((512, 512), Image.Resampling.LANCZOS)

    test_images.extend([np.array(img_me), np.array(img_not_me)])


for index, img in enumerate(test_images):
    test_images[index] = (img, model.predict(img[None, :, :])[0][0], index)


test_images.sort(key=lambda x: x[1])

print(test_images[-1][2] + 1)  # The image that looks the most like me
