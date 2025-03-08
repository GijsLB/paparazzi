import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ===== TUNE THESE =====
NUM_COLUMNS = 200
NUM_BLOCKS_PER_COLUMN = 80
THRESH_OBSTACLE = 0.2
THRESH_EDGE = 0.2
X_WHITE_TILES = 2
Y_BLACK_TILES = 10

# The three images we want to display side by side
image_names = [
    "21382693.jpg",  # "first"  
    "49182469.jpg",
    "39415877.jpg"   # net
]

input_dir = os.path.expanduser("~/paparazzi/prototyping/decision_logic/test_frames")

green_filter = ((75, 250), (110, 155), (50, 145))

def apply_green_filter(image_yuv, y_range, u_range, v_range):
    """Return a binary (255/0) mask of areas considered 'green' under YUV thresholds."""
    mask = (
        (image_yuv[:,:,0] >= y_range[0]) & (image_yuv[:,:,0] <= y_range[1]) &
        (image_yuv[:,:,1] >= u_range[0]) & (image_yuv[:,:,1] <= u_range[1]) &
        (image_yuv[:,:,2] >= v_range[0]) & (image_yuv[:,:,2] <= v_range[1])
    )
    filtered = np.zeros_like(image_yuv[:,:,0])
    filtered[mask] = 255
    return filtered

# Create a figure with 4 rows (processing stages) x 3 columns (images)
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
fig.subplots_adjust(wspace=0.05, hspace=0.2)

for idx, image_name in enumerate(image_names):
    # --- 1) LOAD IMAGE ---
    image_path = os.path.join(input_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}.")

    # --- 2) GREEN FILTER ---
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    gf = apply_green_filter(image_yuv, *green_filter)

    # --- 3) ROTATE & MORPH ---
    image_rot = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gf_rot    = cv2.rotate(gf, cv2.ROTATE_90_COUNTERCLOCKWISE)
    kernel = np.ones((5,5), np.uint8)
    image_closed = cv2.morphologyEx(gf_rot, cv2.MORPH_CLOSE, kernel)

    # --- 4) EDGE DETECTION ---
    edges = cv2.Canny(image_rot, 30, 100)
    edge_density = cv2.blur(edges, (5,5))

    # --- 5) BLOCK PARTITIONING ---
    height, width = image_closed.shape
    col_w = width // NUM_COLUMNS
    blk_h = height // NUM_BLOCKS_PER_COLUMN

    whiteness_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
    edge_matrix      = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))

    for col in range(NUM_COLUMNS):
        for block in range(NUM_BLOCKS_PER_COLUMN):
            x0, y0 = col*col_w, block*blk_h
            x1, y1 = (col+1)*col_w, (block+1)*blk_h
            region = image_closed[y0:y1, x0:x1]
            whiteness_matrix[col, block] = np.sum(region) / (255.0 * region.size)
            e_val = np.sum(edge_density[y0:y1, x0:x1]) / (255.0 * region.size)
            edge_matrix[col, block] = e_val

    # --- 6) PRE-CLASSIFICATION (COLOR + EDGE) ---
    # "white" if whiteness >= THRESH_OBSTACLE & edge < THRESH_EDGE, else "black"
    preclass = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
    for col in range(NUM_COLUMNS):
        for block in range(NUM_BLOCKS_PER_COLUMN):
            if (whiteness_matrix[col, block] >= THRESH_OBSTACLE) and (edge_matrix[col, block] < THRESH_EDGE):
                preclass[col, block] = 1  # ground candidate
            else:
                preclass[col, block] = 0  # obstacle candidate

    # --- 7) BOTTOM-UP RUN LOGIC ---
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

            if preclass[col, block] == 0:
                # black
                black_run_indices.append(block)
                consecutive_black += 1
                consecutive_white = 0
                if consecutive_black >= Y_BLACK_TILES:
                    locked = True
                    for b_idx in black_run_indices:
                        adjusted_whiteness_matrix[col, b_idx] = 0
                    black_run_indices.clear()
            else:
                # white => treat as black unless X in a row
                consecutive_white += 1
                black_run_indices.append(block)
                if consecutive_white >= X_WHITE_TILES:
                    # Flip entire run to white
                    for b_idx in black_run_indices:
                        adjusted_whiteness_matrix[col, b_idx] = 1
                    black_run_indices.clear()
                    consecutive_black = 0
                    consecutive_white = 0
                else:
                    # keep them black for now
                    for b_idx in black_run_indices:
                        adjusted_whiteness_matrix[col, b_idx] = 0

        # leftover black run
        for b_idx in black_run_indices:
            adjusted_whiteness_matrix[col, b_idx] = 0

    # --- 8) COLOR OVERLAY ---
    for col in range(NUM_COLUMNS):
        for block in range(NUM_BLOCKS_PER_COLUMN):
            x0, y0 = col*col_w, block*blk_h
            x1, y1 = (col+1)*col_w, (block+1)*blk_h
            overlay = filtered_image.copy()
            
            if adjusted_whiteness_matrix[col, block] == 1:
                overlay[y0:y1, x0:x1] = (0, 255, 0)    # ground
            else:
                if preclass[col, block] == 0 and whiteness_matrix[col, block] < THRESH_OBSTACLE:
                    overlay[y0:y1, x0:x1] = (0, 0, 0)  # real obstacle
                else:
                    overlay[y0:y1, x0:x1] = (0, 0, 255)# red false obstacle
            cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)

    # --- 9) DISPLAY RESULTS IN SUBPLOTS ---
    # Row 0: Original
    axes[0, idx].imshow(cv2.cvtColor(image_rot, cv2.COLOR_BGR2RGB))
    axes[0, idx].set_title(f"Original: {image_name}")
    axes[0, idx].axis('off')

    # Row 1: Green filtered
    axes[1, idx].imshow(gf_rot, cmap='gray')
    axes[1, idx].set_title("Green Filtered")
    axes[1, idx].axis('off')

    # Row 2: Edge density or edges
    axes[2, idx].imshow(edge_density, cmap='gray')
    axes[2, idx].set_title("Edge Density")
    axes[2, idx].axis('off')

    # Row 3: Final
    axes[3, idx].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    axes[3, idx].set_title("Ground Refined")
    axes[3, idx].axis('off')

plt.show()
