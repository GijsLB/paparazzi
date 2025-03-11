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
X_WHITE_TILES = 1      # Min number of white tiles required to reset obstacle classification
Y_BLACK_TILES = 10      # Number of consecutive black tiles that forces everything above to stay black

# Define image paths
#image_name = "1284881463.jpg"
image_name = "1231615251.jpg"
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

# ==========================
# NEW LOGIC: RECLASSIFY BLACK BLOBS
# THAT DON'T TOUCH BOUNDARY => GROUND
# ==========================
# 1) Invert so black => 255, white => 0
image_closed_inv = cv2.bitwise_not(image_closed)

# 2) Find connected components in the inverted image
num_labels, labels = cv2.connectedComponents(image_closed_inv)

# 3) Identify labels that touch the boundary => remain black
height, width = image_closed.shape
boundary_labels = set()

# Add top/bottom row
for x in range(width):
    boundary_labels.add(labels[0, x])
    boundary_labels.add(labels[height - 1, x])

# Add left/right column
for y in range(height):
    boundary_labels.add(labels[y, 0])
    boundary_labels.add(labels[y, width - 1])

# 4) For any label NOT in boundary => set those pixels to white in image_closed
for label_idx in range(1, num_labels):
    if label_idx not in boundary_labels:
        image_closed[labels == label_idx] = 255

# ================
# BLOCK-BASED LOGIC
# ================
column_width = width // NUM_COLUMNS
block_height = height // NUM_BLOCKS_PER_COLUMN

# Compute whiteness per block
whiteness_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
for col in range(NUM_COLUMNS):
    for block in range(NUM_BLOCKS_PER_COLUMN):
        x_start = col * column_width
        y_start = block * block_height
        x_end   = (col + 1) * column_width
        y_end   = (block + 1) * block_height
        block_region = image_closed[y_start:y_end, x_start:x_end]
        # Each block's whiteness fraction
        whiteness_matrix[col, block] = np.sum(block_region) / (255.0 * block_region.size)

# Bottom-up obstacle logic
adjusted_whiteness_matrix = whiteness_matrix.copy()

filtered_image = cv2.cvtColor(image_closed, cv2.COLOR_GRAY2BGR)

for col in range(NUM_COLUMNS):
    locked = False
    consecutive_black = 0
    consecutive_white = 0
    
    # Keep track of the "current black run"
    black_run_indices = []
    
    for block in reversed(range(NUM_BLOCKS_PER_COLUMN)):
        if locked:
            # Once locked, force everything above to black
            adjusted_whiteness_matrix[col, block] = 0
            continue
        
        # Check if the tile is below obstacle threshold
        is_black_tile = (whiteness_matrix[col, block] < THRESH_OBSTACLE)
        
        if is_black_tile:
            black_run_indices.append(block)
            consecutive_black += 1
            consecutive_white = 0

            # If we’ve now reached Y consecutive black tiles => lock
            if consecutive_black >= Y_BLACK_TILES:
                locked = True
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0
                black_run_indices.clear()

        else:
            # We see a white tile. Treat it “as black” unless/until we see X consecutive white
            consecutive_white += 1
            black_run_indices.append(block)
            
            if consecutive_white >= X_WHITE_TILES:
                # Retroactively flip the entire current run to white
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 1
                black_run_indices.clear()
                consecutive_black = 0
                consecutive_white = 0
            else:
                # Not enough consecutive whites => keep them black
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0

    # If we finish the column with leftover black_run_indices => black
    for b_idx in black_run_indices:
        adjusted_whiteness_matrix[col, b_idx] = 0

# ------------------
# COLOR OVERLAY SECTION
# ------------------
for col in range(NUM_COLUMNS):
    for block in range(NUM_BLOCKS_PER_COLUMN):
        x_start = col * column_width
        y_start = block * block_height
        x_end   = min((col + 1) * column_width, width)
        y_end   = (block + 1) * block_height
        overlay = filtered_image.copy()
        
        if adjusted_whiteness_matrix[col, block] == 1:
            # Ground -> green
            overlay[y_start:y_end, x_start:x_end] = (0, 255, 0)
        else:
            # Obstacle -> check if below threshold
            if whiteness_matrix[col, block] < THRESH_OBSTACLE:
                overlay[y_start:y_end, x_start:x_end] = (0, 0, 0)   # real obstacle
            else:
                overlay[y_start:y_end, x_start:x_end] = (0, 0, 255) # false obstacle
        
        cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)

# Compute whiteness arrays for plotting
whiteness_array = 1 - np.mean(whiteness_matrix, axis=1)
adjusted_whiteness = 1 - np.mean(adjusted_whiteness_matrix, axis=1)

# Print them before display
print("Original Whiteness Array:", whiteness_array)
print("Adjusted Whiteness Array:", adjusted_whiteness)

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")

axes[1].imshow(image_closed_inv, cmap="gray")
axes[1].set_title("Inverted Image (for Black-Blob Connectivity)")

axes[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
axes[2].plot(np.linspace(0, width, NUM_COLUMNS), adjusted_whiteness * height, color="blue", linewidth=2)
axes[2].set_title("Ground Refined Image (Bottom-Up)")

plt.tight_layout()
plt.show()
