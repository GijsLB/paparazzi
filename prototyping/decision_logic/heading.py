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

# Function to find the green column with the highest green score in the right half of the image
def find_highest_green_column(image):
    height, width, _ = image.shape

    # Only process the right half of the image
    right_image = image[:, width//2:]

    # Split BGR channels
    blue, green, red = cv2.split(right_image)

    # Calculate the green score for each column in the right half
    green_scores = np.sum(green, axis=0)

    # Find the column with the highest green score
    highest_green_column = np.argmax(green_scores)

    # Adjust the index to refer to the right half of the original image
    highest_green_column += width // 2

    return highest_green_column


# Function to mark the highest green column as blue in the right half
def mark_highest_green_column(image):
    highest_green_column = find_highest_green_column(image)
    height, width, _ = image.shape

    # Mark the highest green column as blue in the right half
    image[:, highest_green_column:highest_green_column+5] = [0, 0, 255]  # Blue in BGR

    return image


# Process each image
new_images = []
for image in images:
    new_image = mark_highest_green_column(image)
    new_images.append(new_image)

# Save the output as a GIF
imageio.mimsave(output_path, new_images, duration=0.3)
