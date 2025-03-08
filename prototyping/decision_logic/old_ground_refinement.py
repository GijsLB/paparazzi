import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt

# Define image paths
image_name = "21382693.jpg"
input_dir = os.path.expanduser("~/paparazzi/prototyping/decision_logic/test_frames")
image_path = os.path.join(input_dir, image_name)

# Load the original image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Error: Could not load image at {image_path}. Check the file path.")

# Convert to YUV and apply green filter
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
green_filter = ((75, 250), (110, 155), (50, 145))

def apply_green_filter(image, y_range, u_range, v_range):
    mask = (image[:, :, 0] >= y_range[0]) & (image[:, :, 0] <= y_range[1]) & \
           (image[:, :, 1] >= u_range[0]) & (image[:, :, 1] <= u_range[1]) & \
           (image[:, :, 2] >= v_range[0]) & (image[:, :, 2] <= v_range[1])
    filtered = np.zeros_like(image[:, :, 0])
    filtered[mask] = 255  # White for detected areas
    return filtered

green_filtered = apply_green_filter(image_yuv, *green_filter)

# Rotate and process the filtered image
image_rotated = cv2.rotate(green_filtered, cv2.ROTATE_90_COUNTERCLOCKWISE)
kernel = np.ones((5,5), np.uint8)
image_closed = cv2.morphologyEx(image_rotated, cv2.MORPH_CLOSE, kernel)

# Define segmentation parameters
num_columns = 50
num_blocks_per_column = 20
height, width = image_closed.shape
column_width = width // num_columns
block_height = height // num_blocks_per_column

# Compute whiteness degree per block
whiteness_matrix = np.zeros((num_columns, num_blocks_per_column))
for col in range(num_columns):
    for block in range(num_blocks_per_column):
        x_start, y_start = col * column_width, block * block_height
        x_end, y_end = (col + 1) * column_width, (block + 1) * block_height
        block_region = image_closed[y_start:y_end, x_start:x_end]
        whiteness_matrix[col, block] = np.sum(block_region) / (255 * block_region.size)

# Apply ground refinement logic
adjusted_whiteness_matrix = whiteness_matrix.copy()
thresh_obstacle = 0.5
filtered_image = cv2.cvtColor(image_rotated, cv2.COLOR_GRAY2BGR)
for col in range(num_columns):
    found_black = False
    consecutive_white = 0
    for block in range(num_blocks_per_column):
        x_start, y_start = col * column_width, block * block_height
        x_end, y_end = (col + 1) * column_width, (block + 1) * block_height
        if whiteness_matrix[col, block] < thresh_obstacle:
            found_black = True
            consecutive_white = 0
        else:
            if found_black:
                consecutive_white += 1
                if consecutive_white >= 3:
                    found_black = False
            if not found_black:
                adjusted_whiteness_matrix[col, block] = 1
        overlay = filtered_image.copy()
        if adjusted_whiteness_matrix[col, block] == 1:
            overlay[y_start:y_end, x_start:x_end] = (0, 255, 0)
        elif whiteness_matrix[col, block] < thresh_obstacle:
            overlay[y_start:y_end, x_start:x_end] = (0, 0, 0)
        else:
            overlay[y_start:y_end, x_start:x_end] = (0, 0, 255)
        cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)

# Edge detection on the original image
edges = cv2.Canny(image, 100, 200)

# Display the results
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[1].imshow(green_filtered, cmap="gray")
axes[1].set_title("Green Filtered Image")
axes[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
axes[2].set_title("Ground Refined Image with Edge Detection")
plt.show()

# Print whiteness array
whiteness_array = 1 - np.mean(whiteness_matrix, axis=1)
adjusted_whiteness = 1 - np.mean(adjusted_whiteness_matrix, axis=1)
print("Original Whiteness Array:", whiteness_array)
print("Adjusted Whiteness Array:", adjusted_whiteness)