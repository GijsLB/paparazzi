import cv2
import numpy as np
import os

# ======================
# CONFIGURABLE VARIABLES
# ======================
NUM_COLUMNS = 100  
NUM_BLOCKS_PER_COLUMN = 40  
THRESH_OBSTACLE = 0.4
THRESH_EDGE = 0.7
X_WHITE_TILES = 1      
Y_BLACK_TILES = 8      

# Define paths
input_dir = os.path.expanduser("~/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt")
output_dir = os.path.join(input_dir, "filtered_images")

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Start processing only after finding this image
process_images = False  
for image_name in sorted(os.listdir(input_dir)):  
    if not image_name.endswith(".jpg"):
        continue  
    
    if image_name == "1143815935.jpg":
        process_images = True  
    
    if not process_images:
        continue  
    
    print(f"Processing {image_name}...")

    image_path = os.path.join(input_dir, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_name}, could not be loaded.")
        continue

    # === APPLY FILTERS ===
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    green_filter = ((90, 210), (75, 115), (69, 145))

    def apply_green_filter(image, y_range, u_range, v_range):
        mask = (
            (image[:, :, 0] >= y_range[0]) & (image[:, :, 0] <= y_range[1]) &
            (image[:, :, 1] >= u_range[0]) & (image[:, :, 1] <= u_range[1]) &
            (image[:, :, 2] >= v_range[0]) & (image[:, :, 2] <= v_range[1])
        )
        filtered = np.zeros_like(image[:, :, 0])
        filtered[mask] = 255  
        return filtered

    green_filtered = apply_green_filter(image_yuv, *green_filter)

    # Rotate images counterclockwise
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    green_filtered = cv2.rotate(green_filtered, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Apply morphological closing
    kernel = np.ones((5,5), np.uint8)
    image_closed = cv2.morphologyEx(green_filtered, cv2.MORPH_CLOSE, kernel)

    # Edge detection
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
            x_end, y_end = min((col + 1) * column_width, width), (block + 1) * block_height  # Ensure last column is processed
            block_region = image_closed[y_start:y_end, x_start:x_end]
            whiteness_matrix[col, block] = np.sum(block_region) / (255.0 * block_region.size)

    # Bottom-up obstacle logic
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

    # COLOR OVERLAY
    for col in range(NUM_COLUMNS):
        for block in range(NUM_BLOCKS_PER_COLUMN):
            x_start, y_start = col * column_width, block * block_height
            x_end, y_end = min((col + 1) * column_width, width), (block + 1) * block_height
            overlay = filtered_image.copy()
            
            if adjusted_whiteness_matrix[col, block] == 1:
                overlay[y_start:y_end, x_start:x_end] = (0, 255, 0)    # ground
            else:
                if whiteness_matrix[col, block] < THRESH_OBSTACLE:
                    overlay[y_start:y_end, x_start:x_end] = (0, 0, 0)   # real obstacle
                else:
                    overlay[y_start:y_end, x_start:x_end] = (0, 0, 255) # false obstacle

            cv2.addWeighted(overlay, 0.7, filtered_image, 0.3, 0, filtered_image)

    # **Concatenate Original & Filtered Side by Side**
    output_combined = np.hstack((image, filtered_image))

    # Save processed image
    filtered_image_name = os.path.join(output_dir, image_name.replace(".jpg", "_filtered.jpg"))
    cv2.imwrite(filtered_image_name, output_combined)

    print(f"Saved: {filtered_image_name}")

print("Processing complete.")
