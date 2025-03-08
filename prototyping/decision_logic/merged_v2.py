import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ===== TUNE THESE =====
NUM_COLUMNS = 100  # Number of vertical columns to divide the image
NUM_BLOCKS_PER_COLUMN = 40  # Number of horizontal blocks per column
THRESH_OBSTACLE = 0.5  # Whiteness threshold to determine obstacles
THRESH_EDGE = 0.1      # Not used directly below, but kept for reference
X_WHITE_TILES = 2      # Min number of white tiles required to reset obstacle classification
Y_BLACK_TILES = 10      # Number of consecutive black tiles that forces everything above to stay black


# -- Load image paths --
image_name = "21382693.jpg" #first
image_name = "49182469.jpg" 
image_name = "39415877.jpg" #net

input_dir = os.path.expanduser("~/paparazzi/prototyping/decision_logic/test_frames")
image_path = os.path.join(input_dir, image_name)

# -- Read & Pre-process image --
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}.")

image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
green_filter = ((75, 250), (110, 155), (50, 145))

def apply_green_filter(image, y_range, u_range, v_range):
    mask = (
        (image[:,:,0] >= y_range[0]) & (image[:,:,0] <= y_range[1]) &
        (image[:,:,1] >= u_range[0]) & (image[:,:,1] <= u_range[1]) &
        (image[:,:,2] >= v_range[0]) & (image[:,:,2] <= v_range[1])
    )
    filtered = np.zeros_like(image[:,:,0])
    filtered[mask] = 255
    return filtered

green_filtered = apply_green_filter(image_yuv, *green_filter)

# Rotate
image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
green_filtered = cv2.rotate(green_filtered, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Morph close
kernel = np.ones((5,5), np.uint8)
image_closed = cv2.morphologyEx(green_filtered, cv2.MORPH_CLOSE, kernel)

# Edges & Edge Density
edges = cv2.Canny(image, 30, 100)
edge_density = cv2.blur(edges, (5,5))

height, width = image_closed.shape
col_w = width // NUM_COLUMNS
blk_h = height // NUM_BLOCKS_PER_COLUMN

# Compute whiteness & edge density
whiteness_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
edge_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))

for col in range(NUM_COLUMNS):
    for block in range(NUM_BLOCKS_PER_COLUMN):
        x0, y0 = col*col_w, block*blk_h
        x1, y1 = (col+1)*col_w, (block+1)*blk_h
        region = image_closed[y0:y1, x0:x1]
        whiteness_matrix[col, block] = np.sum(region)/(255.0*region.size)
        edge_val = np.sum(edge_density[y0:y1, x0:x1])/(255.0*region.size)
        edge_matrix[col, block] = edge_val

# Pre-classify each block:
# "white" if whiteness >= THRESH_OBSTACLE & edge < THRESH_EDGE, else "black"
preclass = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
for col in range(NUM_COLUMNS):
    for block in range(NUM_BLOCKS_PER_COLUMN):
        if (whiteness_matrix[col, block] >= THRESH_OBSTACLE) and (edge_matrix[col, block] < THRESH_EDGE):
            preclass[col, block] = 1  # ground candidate
        else:
            preclass[col, block] = 0  # obstacle candidate

# Bottom-up run logic on preclass
adjusted_whiteness_matrix = preclass.copy()
filtered_image = cv2.cvtColor(image_closed, cv2.COLOR_GRAY2BGR)

for col in range(NUM_COLUMNS):
    locked = False
    consecutive_black = 0
    consecutive_white = 0
    black_run_indices = []

    for block in reversed(range(NUM_BLOCKS_PER_COLUMN)):
        if locked:
            adjusted_whiteness_matrix[col, block] = 0
            continue

        if preclass[col, block] == 0:  # "black"
            black_run_indices.append(block)
            consecutive_black += 1
            consecutive_white = 0
            if consecutive_black >= Y_BLACK_TILES:
                locked = True
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0
                black_run_indices.clear()
        else:
            # White tile - treat it as black until X consecutive
            consecutive_white += 1
            black_run_indices.append(block)
            if consecutive_white >= X_WHITE_TILES:
                # Flip the run to white
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 1
                black_run_indices.clear()
                consecutive_black = 0
                consecutive_white = 0
            else:
                # Keep them black for now
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0

    # leftover black run
    for b_idx in black_run_indices:
        adjusted_whiteness_matrix[col, b_idx] = 0

# Color overlay
for col in range(NUM_COLUMNS):
    for block in range(NUM_BLOCKS_PER_COLUMN):
        x0, y0 = col*col_w, block*blk_h
        x1, y1 = (col+1)*col_w, (block+1)*blk_h
        overlay = filtered_image.copy()
        
        if adjusted_whiteness_matrix[col, block] == 1:
            overlay[y0:y1, x0:x1] = (0, 255, 0)    # ground
        else:
            # If originally below whiteness threshold or high edges => real obstacle
            if preclass[col, block] == 0 and whiteness_matrix[col, block] < THRESH_OBSTACLE:
                overlay[y0:y1, x0:x1] = (0, 0, 0)
            else:
                overlay[y0:y1, x0:x1] = (0, 0, 255) # red false obstacle
        cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)

# Plot
whiteness_array = 1 - np.mean(whiteness_matrix, axis=1)
adjusted_whiteness = 1 - np.mean(adjusted_whiteness_matrix, axis=1)
print("Original Whiteness Array:", whiteness_array)
print("Adjusted Whiteness Array:", adjusted_whiteness)

fig, axes = plt.subplots(4, 1, figsize=(10,20))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[1].imshow(green_filtered, cmap="gray")
axes[1].set_title("Green Filtered Image")
axes[2].imshow(edge_density, cmap="gray")
axes[2].set_title("Edge Density (Blurred)")
axes[3].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
axes[3].plot(np.linspace(0, width, NUM_COLUMNS), adjusted_whiteness*height, color="blue", linewidth=2)
axes[3].set_title("Solution 2: Simple Color + Global Edge Threshold")
plt.show()