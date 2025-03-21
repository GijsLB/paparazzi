import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt

# ======================
# CONFIGURABLE VARIABLES
# ======================
NUM_COLUMNS = 100
NUM_BLOCKS_PER_COLUMN = 40
THRESH_OBSTACLE = 0.4

# Suppose you want the threshold line at 50% image height
HEIGHT_TRANSITION_FRACTION = 0.3

# “Lenient” vs. “Strict” sets
# lenient => we use small X_WHITE (so fewer whites required) & large Y_BLACK
X_WHITE_LENIENT = 1
Y_BLACK_LENIENT = 11

# strict => bigger X_WHITE, smaller Y_BLACK
X_WHITE_STRICT = 1
Y_BLACK_STRICT = 3

# Define image paths
# image_name = "1151182529.jpg" 
image_name = "1284881463.jpg"
# image_name = "1231615251.jpg"
input_dir = os.path.expanduser("~/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt")
image_path = os.path.join(input_dir, image_name)

# Load the original image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Error: Could not load image at {image_path}")

# Convert to YUV and apply green filter
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
green_filter = ((90, 210), (75, 115), (69, 145))

def apply_green_filter(image, y_range, u_range, v_range):
    mask = (
        (image[:, :, 0] >= y_range[0]) & (image[:, :, 0] <= y_range[1]) &
        (image[:, :, 1] >= u_range[0]) & (image[:, :, 1] <= u_range[1]) &
        (image[:, :, 2] >= v_range[0]) & (image[:, :, 2] <= v_range[1])
    )
    out = np.zeros_like(image[:, :, 0])
    out[mask] = 255
    return out

green_filtered = apply_green_filter(image_yuv, *green_filter)

# Rotate images counterclockwise
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
green_filtered = cv2.rotate(green_filtered, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Morphological closing
kernel = np.ones((5,5), np.uint8)
image_closed = cv2.morphologyEx(green_filtered, cv2.MORPH_CLOSE, kernel)

# Segmentation
height, width = image_closed.shape
column_width  = width // NUM_COLUMNS
block_height  = height // NUM_BLOCKS_PER_COLUMN

transition_line = HEIGHT_TRANSITION_FRACTION * height  # e.g. 0.5 * height

# Compute whiteness per block
whiteness_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
for col in range(NUM_COLUMNS):
    for blk in range(NUM_BLOCKS_PER_COLUMN):
        x_start = col * column_width
        y_start = blk * block_height
        x_end   = (col + 1) * column_width
        y_end   = (blk + 1) * block_height
        block_region = image_closed[y_start:y_end, x_start:x_end]
        whiteness_matrix[col, blk] = np.sum(block_region) / (255.0 * block_region.size)

# Bottom-up logic with one-time switch from lenient to strict
adjusted_whiteness_matrix = whiteness_matrix.copy()
filtered_image = cv2.cvtColor(image_closed, cv2.COLOR_GRAY2BGR)

for col in range(NUM_COLUMNS):
    locked = False
    consecutive_black = 0
    consecutive_white = 0
    black_run_indices = []

    # We start each column in “lenient” mode
    in_strict_mode = False

    for blk in reversed(range(NUM_BLOCKS_PER_COLUMN)):
        if locked:
            # Once locked, everything above forced black
            adjusted_whiteness_matrix[col, blk] = 0
            continue
        
        # 1) Check if we should switch to strict mode
        block_center_y = (blk + 0.5) * block_height
        if (not in_strict_mode) and (block_center_y < transition_line):
            # The moment we encounter a block whose center is above the threshold line,
            # we set in_strict_mode = True for all subsequent blocks (no reset of counters)
            in_strict_mode = True

        # 2) Pick the X/Y for this row
        if in_strict_mode:
            X_WHITE_TILES = X_WHITE_STRICT
            Y_BLACK_TILES = Y_BLACK_STRICT
        else:
            X_WHITE_TILES = X_WHITE_LENIENT
            Y_BLACK_TILES = Y_BLACK_LENIENT

        # 3) Now do the usual bottom-up logic
        is_black_tile = (whiteness_matrix[col, blk] < THRESH_OBSTACLE)
        if is_black_tile:
            black_run_indices.append(blk)
            consecutive_black += 1
            consecutive_white = 0

            if consecutive_black >= Y_BLACK_TILES:
                locked = True
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0
                black_run_indices.clear()

        else:
            consecutive_white += 1
            black_run_indices.append(blk)

            if consecutive_white >= X_WHITE_TILES:
                # Flip run to white
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 1
                black_run_indices.clear()
                consecutive_black = 0
                consecutive_white = 0
            else:
                # Not enough whites => keep them black
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0

    # leftover => black
    for b_idx in black_run_indices:
        adjusted_whiteness_matrix[col, b_idx] = 0

# Initialize an array to store the count of green blocks per column
green_blocks_per_column = np.zeros(NUM_COLUMNS)

# Iterate through each column to count the number of green blocks
for col in range(NUM_COLUMNS):
    green_block_count = 0  # Reset count for each column
    
    for blk in range(NUM_BLOCKS_PER_COLUMN):
        x_start = col * column_width
        y_start = blk * block_height
        x_end = (col + 1) * column_width
        y_end = (blk + 1) * block_height
        block_region = green_filtered[y_start:y_end, x_start:x_end]
        overlay = filtered_image.copy()

        if adjusted_whiteness_matrix[col, blk] == 1:
            overlay[y_start:y_end, x_start:x_end] = (0,255,0)
        else:
            if whiteness_matrix[col, blk] < THRESH_OBSTACLE:
                overlay[y_start:y_end, x_start:x_end] = (0,0,0)
            else:
                overlay[y_start:y_end, x_start:x_end] = (0,0,255)
        
        #cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)
        filtered_image = overlay.copy()
        
        # Check if the block is green (non-zero in the green channel)
        if np.sum(block_region) > 0:  # If there's green intensity in the block
            green_block_count += 1

    # Store the count for the current column
    green_blocks_per_column[col] = green_block_count

best_index = 0
best_count = 0
for i in range(len(green_blocks_per_column)):
    if green_blocks_per_column[i] > best_count:
        best_index = i 
        best_count = green_blocks_per_column[i]
print(best_index, best_count)

# Print the array containing the number of green blocks per column
print("Green Blocks Per Column:", green_blocks_per_column)

for blk in range(NUM_BLOCKS_PER_COLUMN):
    x_start = best_index * column_width
    y_start = blk * block_height
    x_end = (best_index + 1) * column_width
    y_end = (blk + 1) * block_height
    
    # Apply blue color to all blocks in the column with the highest green score
    filtered_image[y_start:y_end, x_start:x_end] = (255, 0, 0)  # Blue color (BGR)

# For debugging
whiteness_array = 1 - np.mean(whiteness_matrix, axis=1)
adjusted_whiteness = 1 - np.mean(adjusted_whiteness_matrix, axis=1)

# print("Original Whiteness Array:", whiteness_array)
# print("Adjusted Whiteness Array:", adjusted_whiteness)

# Visualization
fig, axes = plt.subplots(3, 1, figsize=(10,15))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")

axes[1].imshow(green_filtered, cmap="gray")
axes[1].set_title("Green Filtered")

axes[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
axes[2].set_title("Refined (Lenient -> Strict No Reset)")

plt.tight_layout()
plt.show()
