import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt

# ======================
# CONFIGURABLE VARIABLES
# ======================
NUM_COLUMNS = 85 #520
NUM_BLOCKS_PER_COLUMN = 40 #240
THRESH_OBSTACLE = 0.4

# Suppose you want the threshold line at 50% image height
HEIGHT_TRANSITION_FRACTION = 0.3

# “Lenient” vs. “Strict” sets
# lenient => we use small X_WHITE (so fewer whites required) & large Y_BLACK
X_WHITE_LENIENT = 1
Y_BLACK_LENIENT = 8

# strict => bigger X_WHITE, smaller Y_BLACK
X_WHITE_STRICT = 1
Y_BLACK_STRICT = Y_BLACK_LENIENT

# Define image paths
image_name = "1284881463.jpg"
image_name = "1231615251.jpg"
# image_name = "1169949043.jpg"

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

image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
green_filtered = cv2.rotate(green_filtered, cv2.ROTATE_90_COUNTERCLOCKWISE)

kernel = np.ones((5,5), np.uint8)
image_closed = cv2.morphologyEx(green_filtered, cv2.MORPH_CLOSE, kernel)

height, width = image_closed.shape
column_width  = width // NUM_COLUMNS
block_height  = height // NUM_BLOCKS_PER_COLUMN

transition_line = HEIGHT_TRANSITION_FRACTION * height

whiteness_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
was_flipped = np.zeros_like(whiteness_matrix, dtype=bool)
for col in range(NUM_COLUMNS):
    for blk in range(NUM_BLOCKS_PER_COLUMN):
        x_start = col * column_width
        y_start = blk * block_height
        x_end   = (col + 1) * column_width
        y_end   = (blk + 1) * block_height
        block_region = image_closed[y_start:y_end, x_start:x_end]
        whiteness_matrix[col, blk] = np.sum(block_region) / (255.0 * block_region.size)

adjusted_whiteness_matrix = whiteness_matrix.copy()
filtered_image = cv2.cvtColor(image_closed, cv2.COLOR_GRAY2BGR)

for col in range(NUM_COLUMNS):
    locked = False
    consecutive_black = 0
    consecutive_white = 0
    black_run_indices = []
    in_strict_mode = False

    for blk in reversed(range(NUM_BLOCKS_PER_COLUMN)):
        if locked:
            adjusted_whiteness_matrix[col, blk] = 0
            continue

        block_center_y = (blk + 0.5) * block_height
        if (not in_strict_mode) and (block_center_y < transition_line):
            in_strict_mode = True

        if in_strict_mode:
            X_WHITE_TILES = X_WHITE_STRICT
            Y_BLACK_TILES = Y_BLACK_STRICT
        else:
            X_WHITE_TILES = X_WHITE_LENIENT
            Y_BLACK_TILES = Y_BLACK_LENIENT

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
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 1
                    if whiteness_matrix[col, b_idx] < THRESH_OBSTACLE:
                        was_flipped[col, b_idx] = True  # mark flipped black→white
                black_run_indices.clear()
                consecutive_black = 0
                consecutive_white = 0
            else:
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0

    for b_idx in black_run_indices:
        adjusted_whiteness_matrix[col, b_idx] = 0

for col in range(NUM_COLUMNS):
    for blk in range(NUM_BLOCKS_PER_COLUMN):
        x_start = col * column_width
        y_start = blk * block_height
        x_end   = min((col + 1) * column_width, width)
        y_end   = min((blk + 1) * block_height, height)
        overlay = filtered_image.copy()

        if adjusted_whiteness_matrix[col, blk] == 1:
            if was_flipped[col, blk]:
                overlay[y_start:y_end, x_start:x_end] = (0, 100, 0)  # dark green for flipped
            else:
                overlay[y_start:y_end, x_start:x_end] = (0, 255, 0)  # normal green
        else:
            if whiteness_matrix[col, blk] < THRESH_OBSTACLE:
                overlay[y_start:y_end, x_start:x_end] = (0,0,0)
            else:
                overlay[y_start:y_end, x_start:x_end] = (0,0,255)
        
        filtered_image = overlay.copy()

whiteness_array = np.sum(whiteness_matrix, axis=1).astype(int)
adjusted_whiteness = np.sum(adjusted_whiteness_matrix, axis=1).astype(int)

print("Original Whiteness Array:", whiteness_array)
print("Adjusted Whiteness Array:", adjusted_whiteness)

fig, axes = plt.subplots(3, 1, figsize=(10,15))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")

axes[1].imshow(green_filtered, cmap="gray")
axes[1].set_title("Green Filtered")

axes[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
axes[2].set_title("Refined (Dark Green = Reclassified from Black)")

plt.tight_layout()
plt.show()

