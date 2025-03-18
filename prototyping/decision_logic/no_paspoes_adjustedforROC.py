import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt



def process_image(NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN, HEIGHT_TRANSITION_FRACTION, 
                  X_WHITE_LENIENT, Y_BLACK_LENIENT, X_WHITE_STRICT, Y_BLACK_STRICT, 
                  THRESH_OBSTACLE, image_path, widthGT, heightGT, aaa):


    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Could not load image at {image_path}")
    

    # Convert to YUV and apply green filter
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    green_filter = ((aaa, 210), (75, 115), (69, 145))

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

    adjusted_whiteness_matrix = cv2.rotate(adjusted_whiteness_matrix, cv2.ROTATE_90_COUNTERCLOCKWISE)
    adjusted_whiteness_matrix = cv2.flip(adjusted_whiteness_matrix, 0) 


    resized = cv2.resize(adjusted_whiteness_matrix, 
        (heightGT, widthGT), 
        interpolation=cv2.INTER_NEAREST)


    
    return resized

# NUM_COLUMNS = 100
# NUM_BLOCKS_PER_COLUMN = 40
# # Suppose you want the threshold line at 50% image height
# HEIGHT_TRANSITION_FRACTION = 0.3
# # “Lenient” vs. “Strict” sets
# # lenient => we use small X_WHITE (so fewer whites required) & large Y_BLACK
# X_WHITE_LENIENT = 1
# Y_BLACK_LENIENT = 11
# # strict => bigger X_WHITE, smaller Y_BLACK
# X_WHITE_STRICT = 1
# Y_BLACK_STRICT = 3

# THRESH_OBSTACLE = 0.4

# image_name = "780418796.jpg"
# input_dir = os.path.expanduser("~/paparazzi/prototyping/Groundtruth/testlabel1")
# image_path = os.path.join(input_dir, image_name)

# whitetest = process_image(NUM_COLUMNS, NUM_BLOCKS_PER_COLUMN, HEIGHT_TRANSITION_FRACTION,
#                                    X_WHITE_LENIENT, Y_BLACK_LENIENT, X_WHITE_STRICT, Y_BLACK_STRICT,
#                                    THRESH_OBSTACLE, image_path, 360, 720)

# plt.figure()
# plt.imshow(whitetest, cmap='gray')
# plt.show()

