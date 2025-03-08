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
Y_BLACK_TILES = 8      # Number of consecutive black tiles that forces everything above to stay black

MAX_STRIP_WIDTH = 1    # Max width for a "thin" strip of columns
EDGE_DIFF = 1          # The difference in ground–obstacle edge to neighbors
TOP_HALF_LIMIT = NUM_BLOCKS_PER_COLUMN // 2  # "top half" boundary

# Define image paths
image_name = "39415877.jpg" # net
image_name = "21382693.jpg" # first
image_name = "49182469.jpg"

input_dir = os.path.expanduser("~/paparazzi/prototyping/decision_logic/test_frames")
image_path = os.path.join(input_dir, image_name)

# Load the original image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Could not load image at {image_path}.")

# Convert to YUV and apply green filter
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
green_filter = ((75, 250), (110, 155), (50, 145))

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

# Edge detection (just for display)
edges = cv2.Canny(image, 30, 100)
edge_density = cv2.blur(edges, (5,5))

# Define segmentation parameters
height, width = image_closed.shape
column_width = width // NUM_COLUMNS
block_height = height // NUM_BLOCKS_PER_COLUMN

# Compute whiteness per block
whiteness_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
for col in range(NUM_COLUMNS):
    for block in range(NUM_BLOCKS_PER_COLUMN):
        x_start = col * column_width
        x_end   = (col + 1) * column_width
        y_start = block * block_height
        y_end   = (block + 1) * block_height
        block_region = image_closed[y_start:y_end, x_start:x_end]
        whiteness_matrix[col, block] = np.sum(block_region) / (255.0 * block_region.size)

# ===========================
# Bottom-Up Logic (as before)
# ===========================
adjusted_whiteness_matrix = whiteness_matrix.copy()
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

        is_black_tile = (whiteness_matrix[col, block] < THRESH_OBSTACLE)

        if is_black_tile:
            black_run_indices.append(block)
            consecutive_black += 1
            consecutive_white = 0
            if consecutive_black >= Y_BLACK_TILES:
                locked = True
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0
                black_run_indices.clear()
        else:
            consecutive_white += 1
            black_run_indices.append(block)
            if consecutive_white >= X_WHITE_TILES:
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 1
                black_run_indices.clear()
                consecutive_black = 0
                consecutive_white = 0
            else:
                for b_idx in black_run_indices:
                    adjusted_whiteness_matrix[col, b_idx] = 0

    for b_idx in black_run_indices:
        adjusted_whiteness_matrix[col, b_idx] = 0

# ==================================
# HELPER: Find ground–obstacle edge
# ==================================
def get_top_ground_block(col):
    """
    Returns the topmost block index that is ground (1) in this column,
    or None if the column is all obstacle (0).
    """
    for b in range(NUM_BLOCKS_PER_COLUMN):
        if adjusted_whiteness_matrix[col, b] == 1:
            return b
    return None  # No ground

def get_bottom_ground_block(col):
    """
    Returns the bottommost block index that is ground (1) in this column,
    or None if the column is all obstacle (0).
    """
    for b in reversed(range(NUM_BLOCKS_PER_COLUMN)):
        if adjusted_whiteness_matrix[col, b] == 1:
            return b
    return None  # No ground

# =================================================
# 1) Detect "thin false ground" in top half
#    Criteria:
#      - Up to MAX_STRIP_WIDTH columns wide
#      - The top_ground_block differs from neighbors by >= EDGE_DIFF
#      - That top_ground_block is in the top half (b < TOP_HALF_LIMIT)
#    We only COLOR them in YELLOW, do not remove/merge them.
# =================================================
def detect_thin_false_ground_top():
    c = 0
    top_blocks = [get_top_ground_block(col) for col in range(NUM_COLUMNS)]

    thin_columns = []
    while c < NUM_COLUMNS:
        if top_blocks[c] is not None and top_blocks[c] < TOP_HALF_LIMIT:
            # Potentially a ground column in top half
            start = c
            while c < NUM_COLUMNS and top_blocks[c] is not None and top_blocks[c] < TOP_HALF_LIMIT:
                c += 1
            end = c - 1

            run_width = end - start + 1
            if run_width <= MAX_STRIP_WIDTH:
                # check neighbors
                left_col  = start - 1
                right_col = end + 1
                if left_col >= 0 and right_col < NUM_COLUMNS:
                    # difference in top block from neighbors?
                    left_edge  = top_blocks[left_col]
                    right_edge = top_blocks[right_col]
                    if left_edge is not None and right_edge is not None:
                        # If BOTH neighbors have top_ground_block,
                        # check if difference is >= EDGE_DIFF
                        # (Meaning this "thin strip" is quite separate from neighbors)
                        if (abs(top_blocks[start] - left_edge)  >= EDGE_DIFF and
                            abs(top_blocks[end]   - right_edge) >= EDGE_DIFF):
                            # Mark columns [start..end] as thin
                            for cc in range(start, end+1):
                                thin_columns.append(cc)
        else:
            c += 1
    return thin_columns

# =================================================
# 2) Detect "thin false obstacle" in bottom half
#    Criteria:
#      - Up to MAX_STRIP_WIDTH columns
#      - The bottom ground block is significantly different from neighbors
#      - The column is basically obstacle down to near bottom, or has a ground block high up
#      - We only color them YELLOW
# =================================================
def detect_thin_false_obstacle_bottom():
    c = 0
    bottom_blocks = [get_bottom_ground_block(col) for col in range(NUM_COLUMNS)]

    thin_columns = []
    while c < NUM_COLUMNS:
        # bottom_ground_block is None => fully obstacle
        # or if it's up high => effectively obstacle at the bottom
        if bottom_blocks[c] is None or bottom_blocks[c] < (NUM_BLOCKS_PER_COLUMN - 1 - TOP_HALF_LIMIT):
            # Potentially a vertical obstacle near bottom
            start = c
            while c < NUM_COLUMNS and (
                bottom_blocks[c] is None or bottom_blocks[c] < (NUM_BLOCKS_PER_COLUMN - 1 - TOP_HALF_LIMIT)
            ):
                c += 1
            end = c - 1

            run_width = end - start + 1
            if run_width <= MAX_STRIP_WIDTH:
                left_col  = start - 1
                right_col = end + 1
                if left_col >= 0 and right_col < NUM_COLUMNS:
                    left_edge  = bottom_blocks[left_col]
                    right_edge = bottom_blocks[right_col]
                    if left_edge is not None and right_edge is not None:
                        # If neighbors do have ground near bottom
                        if (abs((bottom_blocks[left_col]  or 0) - (bottom_blocks[start] or 0))  >= EDGE_DIFF and
                            abs((bottom_blocks[right_col] or 0) - (bottom_blocks[end]   or 0)) >= EDGE_DIFF):
                            for cc in range(start, end+1):
                                thin_columns.append(cc)
        else:
            c += 1
    return thin_columns

thin_false_ground_cols   = detect_thin_false_ground_top()
thin_false_obstacle_cols = detect_thin_false_obstacle_bottom()

# ================
# COLOR OVERLAY
# ================
for col in range(NUM_COLUMNS):
    x_start = col * column_width
    x_end   = (col + 1) * column_width
    for block in range(NUM_BLOCKS_PER_COLUMN):
        y_start = block * block_height
        y_end   = (block + 1) * block_height

        overlay = filtered_image.copy()
        # The final classification
        val = adjusted_whiteness_matrix[col, block]

        if val == 1:
            # Ground -> green
            overlay[y_start:y_end, x_start:x_end] = (0, 255, 0)
        else:
            # Obstacle -> check original whiteness
            if whiteness_matrix[col, block] < THRESH_OBSTACLE:
                overlay[y_start:y_end, x_start:x_end] = (0, 0, 0)   # real obstacle
            else:
                overlay[y_start:y_end, x_start:x_end] = (0, 0, 255) # false obstacle

        # -------------------------------------
        # Now apply YELLOW if this column is in
        # the "thin false ground" or "thin false obstacle" set
        # -------------------------------------
        if col in thin_false_ground_cols:
            # This is a thin strip in top half => highlight in yellow above ground
            # We'll color from the top down to the topmost ground block
            top_idx = get_top_ground_block(col)
            if top_idx is not None:
                # Blocks from 0..top_idx
                if block <= top_idx:
                    overlay[y_start:y_end, x_start:x_end] = (0, 255, 255)  # yellow
        if col in thin_false_obstacle_cols:
            # This is a thin strip in bottom half => highlight in yellow below ground
            bottom_idx = get_bottom_ground_block(col)
            if bottom_idx is not None:
                # If there's some ground, color from bottom_idx+1 downward
                if block >= bottom_idx:
                    pass  # That part is ground, do nothing
                else:
                    # color this block in yellow
                    overlay[y_start:y_end, x_start:x_end] = (0, 255, 255)
            else:
                # If no ground at all => entire column is obstacle => color in yellow
                overlay[y_start:y_end, x_start:x_end] = (0, 255, 255)

        # Blend overlay
        cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)

# Compute whiteness arrays for plotting
whiteness_array = 1 - np.mean(whiteness_matrix, axis=1)
adjusted_whiteness = 1 - np.mean(adjusted_whiteness_matrix, axis=1)

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
axes[3].set_title("Ground Refined + Thin Strips Marked in Yellow")

plt.show()
