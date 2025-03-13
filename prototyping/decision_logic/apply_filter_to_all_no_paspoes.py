import cv2
import numpy as np
import os

# ======================
# CONFIGURABLE VARIABLES
# ======================
NUM_COLUMNS = 100  # Number of vertical columns to divide the image
NUM_BLOCKS_PER_COLUMN = 40  # Number of horizontal blocks per column
THRESH_OBSTACLE = 0.4  # Whiteness threshold to determine obstacles
X_WHITE_TILES = 1      # Min number of white tiles required to reset obstacle classification
Y_BLACK_TILES = 8      # Number of consecutive black tiles that forces everything above to stay black

CANNY_LOW = 140
CANNY_HIGH = 150

# Define input / output directories
input_dir = os.path.expanduser("~/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt")
output_dir = os.path.join(input_dir, "filtered2")  # for storing results
os.makedirs(output_dir, exist_ok=True)

green_filter = ((90, 210), (75, 115), (69, 145))

def apply_green_filter(image, y_range, u_range, v_range):
    """Return a mask=255 where (Y,U,V) in specified ranges, else 0."""
    mask = (
        (image[:, :, 0] >= y_range[0]) & (image[:, :, 0] <= y_range[1]) &
        (image[:, :, 1] >= u_range[0]) & (image[:, :, 1] <= u_range[1]) &
        (image[:, :, 2] >= v_range[0]) & (image[:, :, 2] <= v_range[1])
    )
    out = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    out[mask] = 255
    return out

# Loop over all images in input_dir
for fname in sorted(os.listdir(input_dir)):
    if not fname.lower().endswith(".jpg"):
        continue  # skip non-jpg

    image_path = os.path.join(input_dir, fname)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {fname}, could not load.")
        continue
    
    print(f"Processing {fname}...")

    # 1) Convert to YUV & green-filter
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    green_filtered = apply_green_filter(image_yuv, *green_filter)

    # 2) Rotate images counterclockwise
    image_rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    filtered_rotated = cv2.rotate(green_filtered, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 3) Morphological closing
    kernel = np.ones((5,5), np.uint8)
    image_closed = cv2.morphologyEx(filtered_rotated, cv2.MORPH_CLOSE, kernel)

    # 4) Edge detection
    edges = cv2.Canny(image_rotated, CANNY_LOW, CANNY_HIGH)
    edge_density = cv2.blur(edges, (5,5))

    # 5) Define segmentation parameters
    height, width = image_closed.shape
    column_width = width // NUM_COLUMNS
    block_height = height // NUM_BLOCKS_PER_COLUMN

    # 6) Compute whiteness per block
    whiteness_matrix = np.zeros((NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN))
    for col in range(NUM_COLUMNS):
        for blk in range(NUM_BLOCKS_PER_COLUMN):
            x_start = col * column_width
            y_start = blk * block_height
            x_end   = min((col + 1) * column_width, width)
            y_end   = min((blk + 1) * block_height, height)
            
            block_region = image_closed[y_start:y_end, x_start:x_end]
            whiteness_matrix[col, blk] = np.sum(block_region) / (255.0 * block_region.size)

    # 7) Bottom-up logic
    adjusted_whiteness_matrix = whiteness_matrix.copy()
    filtered_image = cv2.cvtColor(image_closed, cv2.COLOR_GRAY2BGR)

    for col in range(NUM_COLUMNS):
        locked = False
        consecutive_black = 0
        consecutive_white = 0
        black_run_indices = []
        
        for blk in reversed(range(NUM_BLOCKS_PER_COLUMN)):
            if locked:
                adjusted_whiteness_matrix[col, blk] = 0
                continue

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
                    # Not enough consecutive whites => keep black
                    for b_idx in black_run_indices:
                        adjusted_whiteness_matrix[col, b_idx] = 0

        # leftover => black
        for b_idx in black_run_indices:
            adjusted_whiteness_matrix[col, b_idx] = 0

    # 8) COLOR OVERLAY
    for col in range(NUM_COLUMNS):
        for blk in range(NUM_BLOCKS_PER_COLUMN):
            x_start = col * column_width
            y_start = blk * block_height
            x_end   = min((col + 1) * column_width, width)
            y_end   = min((blk + 1) * block_height, height)
            overlay = filtered_image.copy()

            if adjusted_whiteness_matrix[col, blk] == 1:
                # Ground -> green
                overlay[y_start:y_end, x_start:x_end] = (0, 255, 0)
            else:
                if whiteness_matrix[col, blk] < THRESH_OBSTACLE:
                    # Real obstacle -> black
                    overlay[y_start:y_end, x_start:x_end] = (0, 0, 0)
                else:
                    # False obstacle -> red
                    overlay[y_start:y_end, x_start:x_end] = (0, 0, 255)

            cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)

    # 9) Create a side-by-side for easy comparison
    #    left = original rotated, right = final classification
    final_output = np.hstack((image_rotated, filtered_image))

    # Save result to output folder
    out_name = os.path.splitext(fname)[0] + "_filtered.jpg"
    out_path = os.path.join(output_dir, out_name)
    cv2.imwrite(out_path, final_output)

    print(f"Saved: {out_path}")

print("Processing complete.")
