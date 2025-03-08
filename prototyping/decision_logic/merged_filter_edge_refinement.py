import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt

# ======================
# CONFIGURABLE VARIABLES
# ======================
NUM_COLUMNS = 100  # Number of vertical columns to divide the image
NUM_BLOCKS_PER_COLUMN = 40  # Number of horizontal blocks per column
THRESH_OBSTACLE = 0.4  # Whiteness threshold to determine obstacles
THRESH_EDGE = 1.0  # Edge detection threshold
X_WHITE_TILES = 1      # Min number of white tiles required to reset obstacle classification
Y_BLACK_TILES = 8      # Number of consecutive black tiles that forces everything above to stay black

# Define image paths

image_name = "1248048448.jpg"

input_dir = os.path.expanduser("~/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt")
image_path = os.path.join(input_dir, image_name)

# Load the original image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Error: Could not load image at {image_path}. Check the file path.")

# Convert to YUV and apply green filter
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
green_filter = ((90, 210), (75, 115), (69, 145))

def apply_green_filter(image, y_range, u_range, v_range):
    mask = (
        (image[:, :, 0] >= y_range[0]) & (image[:, :, 0] <= y_range[1]) &
        (image[:, :, 1] >= u_range[0]) & (image[:, :, 1] <= u_range[1]) &
        (image[:, :, 2] >= v_range[0]) & (image[:, :, 2] <= v_range[1])
    )
    filtered = np.zeros_like(image[:, :, 0])
    filtered[mask] = 255  # White for detected areas
    return filtered

green_filtered = apply_green_filter(image_yuv, *green_filter)

# Rotate images counterclockwise
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
green_filtered = cv2.rotate(green_filtered, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Apply morphological closing
kernel = np.ones((5,5), np.uint8)
image_closed = cv2.morphologyEx(green_filtered, cv2.MORPH_CLOSE, kernel)

# Edge detection (not critical to the bottom-up logic)
edges = cv2.Canny(image, 100, 150)
edge_density = cv2.blur(edges, (5,5))

# Define segmentation parameters
height, width = image_closed.shape
column_width = width // NUM_COLUMNS
block_height = height // NUM_BLOCKS_PER_COLUMN

# Compute whiteness per block
whiteness_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
for col in range(NUM_COLUMNS):
    for block in range(NUM_BLOCKS_PER_COLUMN):
        x_start, y_start = col * column_width, block * block_height
        x_end, y_end = (col + 1) * column_width, (block + 1) * block_height
        block_region = image_closed[y_start:y_end, x_start:x_end]
        # Each block's whiteness fraction
        whiteness_matrix[col, block] = np.sum(block_region) / (255.0 * block_region.size)

# Bottom-up obstacle logic that retroactively flips black run if X consecutive whites are reached.

adjusted_whiteness_matrix = whiteness_matrix.copy()

filtered_image = cv2.cvtColor(image_closed, cv2.COLOR_GRAY2BGR)

for col in range(NUM_COLUMNS):
    locked = False
    consecutive_black = 0
    consecutive_white = 0
    
    # Keep track of the "current black run" by storing block indices
    black_run_indices = []
    
    for block in reversed(range(NUM_BLOCKS_PER_COLUMN)):
        if locked:
            # Once locked, force everything above to black
            adjusted_whiteness_matrix[col, block] = 0
            continue
        
        # Check if the original tile is below obstacle threshold
        is_black_tile = (whiteness_matrix[col, block] < THRESH_OBSTACLE)
        
        if is_black_tile:
            # We encountered another black tile
            black_run_indices.append(block)
            consecutive_black += 1
            consecutive_white = 0

            # If we’ve now reached Y consecutive black tiles => lock
            if consecutive_black >= Y_BLACK_TILES:
                locked = True
                # Mark all in the current run black, and continue upward locked
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0
                black_run_indices.clear()

        else:
            # We see a white tile. Treat it “as black” unless/until we see X consecutive white
            consecutive_white += 1
            
            # Add it into the black run for now
            black_run_indices.append(block)
            
            if consecutive_white >= X_WHITE_TILES:
                # Retroactively flip the entire current run to white
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 1
                # Reset counters and clear the run
                black_run_indices.clear()
                consecutive_black = 0
                consecutive_white = 0
            else:
                # We haven't reached X white in a row, so keep everything black so far
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0

    # If we finish the column without locking or flipping, any leftover black_run_indices
    # remain black:
    for b_idx in black_run_indices:
        adjusted_whiteness_matrix[col, b_idx] = 0

# ------------------
# COLOR OVERLAY SECTION
# ------------------
for col in range(NUM_COLUMNS):
    for block in range(NUM_BLOCKS_PER_COLUMN):
        x_start, y_start = col * column_width, block * block_height
        x_end, y_end = (col + 1) * column_width, (block + 1) * block_height
        x_end = min((col + 1) * column_width, width)  # Ensure it does not exceed image width
        overlay = filtered_image.copy()
        
        if adjusted_whiteness_matrix[col, block] == 1:
            # Ground -> green
            overlay[y_start:y_end, x_start:x_end] = (0, 255, 0)
        else:
            # Obstacle -> check if originally below threshold
            if whiteness_matrix[col, block] < THRESH_OBSTACLE:
                # Real obstacle -> black
                overlay[y_start:y_end, x_start:x_end] = (0, 0, 0)
            else:
                # False obstacle -> red
                overlay[y_start:y_end, x_start:x_end] = (0, 0, 255)
        
        cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)


# Compute whiteness arrays for plotting
whiteness_array = 1 - np.mean(whiteness_matrix, axis=1)
adjusted_whiteness = 1 - np.mean(adjusted_whiteness_matrix, axis=1)

# Print them before display
print("Original Whiteness Array:", whiteness_array)
print("Adjusted Whiteness Array:", adjusted_whiteness)

# Visualize
fig, axes = plt.subplots(4, 1, figsize=(10, 20))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[1].imshow(green_filtered, cmap="gray")
axes[1].set_title("Green Filtered Image")
axes[2].imshow(edges, cmap="gray")
axes[2].set_title("Edge Detection Image")
axes[3].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
axes[3].plot(np.linspace(0, width, NUM_COLUMNS), adjusted_whiteness * height, color="blue", linewidth=2)
axes[3].set_title("Ground Refined Image (Bottom-Up) with Color-Coded Tiles")
plt.show()
