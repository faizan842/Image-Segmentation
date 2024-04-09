import pixellib
from pixellib.instance import instance_segmentation
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model for vitamin D kit detection
vitamin_d_model = load_model("vitamin_d_kit_detection_model.h5")

# Function to predict if a vitamin D kit is found in the image
def predict_vitamin_d(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image

    prediction = vitamin_d_model.predict(img)
    if prediction[0][0] > 0.5:
        return True  # Vitamin D kit found
    else:
        return False  # Vitamin D kit not found

# Perform instance segmentation
# segment_image = instance_segmentation()
# segment_image.load_model("vitamin_d_detection_model.h5")
# segment_image.segmentImage("3.jpg", show_bboxes=True, output_image_name="result.jpg")

# Check if a vitamin D kit is found and print the result
if predict_vitamin_d("1.jpg"):
    print("Vitamin D kit found in the image.")
else:
    print("Vitamin D kit not found in the image.")
