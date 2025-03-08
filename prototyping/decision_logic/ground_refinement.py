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

# Apply morphological closing to reduce noise and fill gaps
kernel = np.ones((5,5), np.uint8)  # Kernel size can be tuned
image_closed = cv2.morphologyEx(image_rotated, cv2.MORPH_CLOSE, kernel)

# Define parameters for column and block segmentation
num_columns = 50  # Number of columns
num_blocks_per_column = 20  # Number of blocks per column

height, width = image_closed.shape
column_width = width // num_columns
block_height = height // num_blocks_per_column

# Compute whiteness degree per block
whiteness_matrix = np.zeros((num_columns, num_blocks_per_column))

for col in range(num_columns):
    for block in range(num_blocks_per_column):
        x_start = col * column_width
        y_start = block * block_height
        x_end = (col + 1) * column_width
        y_end = (block + 1) * block_height

        block_region = image_closed[y_start:y_end, x_start:x_end]
        whiteness_matrix[col, block] = np.sum(block_region) / (255 * block_region.size)  # Normalize whiteness

# Apply the vertical filtering logic per block
adjusted_whiteness_matrix = whiteness_matrix.copy()
thresh_obstacle = 0.35  # Lower threshold to better recognize carpets as ground
filtered_image = cv2.cvtColor(image_rotated, cv2.COLOR_GRAY2BGR)  # Convert to color image

for col in range(num_columns):
    found_black = False
    consecutive_white = 0
    for block in range(num_blocks_per_column):
        x_start = col * column_width
        y_start = block * block_height
        x_end = (col + 1) * column_width
        y_end = (block + 1) * block_height

        if whiteness_matrix[col, block] < thresh_obstacle:  # Detected obstacle (black)
            found_black = True
            consecutive_white = 0
        else:  # Detected ground (white or near-white)
            if found_black:
                consecutive_white += 1
                if consecutive_white >= 3:  # Require 3 consecutive white blocks to reclassify obstacle
                    found_black = False
            
            if not found_black:
                adjusted_whiteness_matrix[col, block] = 1  # Reclassify as ground
        
        # Apply color coding
        overlay = filtered_image.copy()
        if adjusted_whiteness_matrix[col, block] == 1:
            overlay[y_start:y_end, x_start:x_end] = (0, 255, 0)  # Green for ground
        elif whiteness_matrix[col, block] < thresh_obstacle:
            overlay[y_start:y_end, x_start:x_end] = (0, 0, 0)  # Black for true obstacle
        else:
            overlay[y_start:y_end, x_start:x_end] = (0, 0, 255)  # Red for false obstacle
        
        cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)  # Apply transparency

# Additional Filtering: Remove Thin Strips of Green (False Ground)
kernel_thin = np.ones((3, 3), np.uint8)  # Small kernel to remove thin structures
filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_OPEN, kernel_thin)

# Connected Component Analysis to Remove Small False Ground Areas
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((adjusted_whiteness_matrix > 0.5).astype(np.uint8))
min_size = 30  # Increase the size threshold for false ground removal
for label in range(1, num_labels):
    if stats[label, cv2.CC_STAT_AREA] < min_size:
        adjusted_whiteness_matrix[labels == label] = 0  # Remove small ground patches

# Remove Ground Detected Too High (Scaffolding Fix)
horizon_threshold = int(height * 0.4)  # Remove any ground classified above 40% of the image height
for col in range(num_columns):
    for block in range(num_blocks_per_column):
        y_start = block * block_height
        if y_start < horizon_threshold:
            adjusted_whiteness_matrix[col, block] = 0  # Force classify as obstacle

# Ensure the rightmost part of the image is processed
filtered_image[filtered_image == 0] = 0  # Ensure all valid regions are updated

# Compute column-wise whiteness for visualization
whiteness_array = 1 - np.mean(whiteness_matrix, axis=1)
adjusted_whiteness = 1 - np.mean(adjusted_whiteness_matrix, axis=1)

# Visualize result using Matplotlib
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Original output
axes[0].imshow(image_rotated, cmap="gray")
axes[0].plot(np.linspace(0, width, num_columns), whiteness_array * height, color="red", linewidth=2)
axes[0].set_title("Original Whiteness Degree Over Image")

# Adjusted output with color overlay
axes[1].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
axes[1].plot(np.linspace(0, width, num_columns), adjusted_whiteness * height, color="blue", linewidth=2)
axes[1].set_title("Adjusted Whiteness Degree After Filtering (Colored)")

# Print the 1D arrays
print("Original Whiteness Array:", whiteness_array)
print("Adjusted Whiteness Array:", adjusted_whiteness)

plt.show()

