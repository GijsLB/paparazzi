"""
    filters.py
    Processes all images in a folder, applies edge detection,
    and saves a side-by-side comparison of the original and detected edges.
"""

import cv2 as cv
import os
import numpy as np

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output directories
image_folder = "/home/sblackmore/Documents/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/"
output_folder = os.path.join(script_dir, "processed_images")

# Get all .jpg images in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Process each image
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Load image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping {image_file}, could not load.")
        continue

    # Rotate image 90° counterclockwise
    img_rotated = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

    # Apply Sobel Y-filter to detect edges
    sobely = cv.Sobel(img_rotated, cv.CV_64F, 1, 0, ksize=5)
    sobelx = cv.Sobel(img_rotated, cv.CV_64F, 0, 1, ksize=5)

    # Detect edge pixels where gradient is strong
    threshold = 700
    edge_pixels = np.column_stack(np.where(sobely > threshold))  # Strong positive edges (red)
    other_edge_pixels = np.column_stack(np.where(sobely < -threshold))  # Strong negative edges (yellow)

    # Convert grayscale image to BGR for color overlay
    img_with_edges = cv.cvtColor(img_rotated, cv.COLOR_GRAY2BGR)

    # Draw red (strong positive) and yellow (strong negative) edge points
    for y, x in edge_pixels:
        cv.circle(img_with_edges, (x, y), 1, (0, 0, 255), -1)  # Red points (BGR: Blue=0, Green=0, Red=255)
    for y, x in other_edge_pixels:
        cv.circle(img_with_edges, (x, y), 1, (0, 255, 255), -1)  # Yellow points (BGR: Blue=0, Green=255, Red=255)

    # Create side-by-side comparison (concatenate images)
    combined_image = np.hstack((cv.cvtColor(img_rotated, cv.COLOR_GRAY2BGR), img_with_edges))

    # Define save path
    save_path = os.path.join(output_folder, f"comparison_{image_file}")

    # Save the side-by-side image
    cv.imwrite(save_path, combined_image)

    print(f"Saved: {save_path}")

