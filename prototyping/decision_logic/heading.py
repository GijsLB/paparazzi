import cv2
import numpy as np
import os
import imageio.v2 as imageio

# Define paths
input_dir = os.path.expanduser("~/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt/filtered_images")
output_dir = os.path.expanduser("~/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt/heading_images")

# Define output GIF
output_gif = "output.gif"
output_path = os.path.join(output_dir, output_gif)

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Pull images
images = []
for item in sorted(os.listdir(input_dir)):
    if not item.endswith(".jpg"):
        continue  
    image_path = os.path.join(input_dir, item)
    image = cv2.imread(image_path)
    images.append(image)

# Detect green in flight path
def green_path(image):
    # Specify region
    x = int(1040 / 4 * 3 - 25)
    y = 230
    height = 10
    width = 50
    roi = image[y:y + height, x:x + width]

    # Split BGR channels
    blue, green, red = cv2.split(roi)

    # Find pixels where green is dominant
    green_threshold = 150
    green_mask = (green > green_threshold) & (green > red) & (green > blue)

    # If any pixel is green, return True
    return np.all(green_mask)


def steering_marker(image, condition):
    x = int(1040 / 4 * 3 - 25)
    y = 0
    height = 10
    width = 50
    if not condition:
        # Paint the region blue (BGR: (255, 0, 0))
        image[y:y + height, x:x + width] = (0, 0, 255)
    return image

new_images = []
for image in images:
    new_image = steering_marker(image, green_path(image))
    new_images.append(new_image)
imageio.mimsave(output_path, new_images, duration=0.1)

   