import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/faizanhabib/Downloads/Cassette Images by Anuja (March 2024)/#1/#1-1800K.JPG')

# Convert to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper bounds for red color

# Threshold the HSV image to get only red colors

# Calculate the average intensity of the red channel in the masked region
red_intensity = np.mean(hsv_image[:, :, 2])

print("Intensity of red color:", red_intensity)
