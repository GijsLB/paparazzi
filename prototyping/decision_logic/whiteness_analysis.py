import cv2
import numpy as np
import os

import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt


# Load the black and white filtered image
image_path = os.path.expanduser("~/paparazzi/prototyping/decision_logic/test_frames/21382693_greenfilter.jpg")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: Could not load image at {image_path}. Check the file path.")

# Rotate the image 90 degrees counterclockwise
image_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Define parameters for column and block segmentation
num_columns = 50  # Number of columns
num_blocks_per_column = 20  # Number of blocks per column

height, width = image_rotated.shape
column_width = width // num_columns
block_height = height // num_blocks_per_column

# Compute whiteness degree per column
whiteness_array = np.zeros(num_columns)

for col in range(num_columns):
    for block in range(num_blocks_per_column):
        x_start = col * column_width
        y_start = block * block_height
        x_end = (col + 1) * column_width
        y_end = (block + 1) * block_height

        block_region = image_rotated[y_start:y_end, x_start:x_end]
        whiteness_array[col] += np.sum(block_region) / 255  # Normalize to count white pixels

# Normalize whiteness array for visualization
whiteness_array = 1 - (whiteness_array / whiteness_array.max())  # Invert values

# Visualize result using Matplotlib
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(image_rotated, cmap="gray")
shift_amount = column_width / 2  # Small shift to the right
column_edges = np.linspace(0, width, num_columns + 1)  # Get edges of columns
column_centers = (column_edges[:-1] + column_edges[1:]) / 2  # Compute exact centers
ax.plot(column_centers - shift_amount, whiteness_array * height, color="red", linewidth=2)
ax.set_title("Whiteness Degree Over Image")
plt.show()

# Print the 1D array
print("Whiteness Array:", whiteness_array)