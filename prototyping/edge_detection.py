‘’’
"""
    Processes all images in a folder using OpenCV's Sobel operator (ksize=3),
    computes the edge ratio for the middle and lower areas, draws the edges
    on a blank output image, and annotates the output to indicate carpet vs obstacle.
"""

import cv2 as cv
import os
import numpy as np

def process_image(image_path, output_path):
    # Edge ratio threshold
    ratio_threshold = 2.

    # Load image in grayscale
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping {image_path}, could not load.")
        return

    # Rotate image 90° counterclockwise
    img_rotated = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
    height, width = img_rotated.shape

    # Compute gradients using OpenCV's Sobel (kernel size=3 gives the same kernels as your C code)
    grad_x = cv.Sobel(img_rotated, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img_rotated, cv.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold to create an edge mask
    edge_threshold = 100
    edge_mask = grad_mag > edge_threshold

    # Count edge pixels in middle and lower thirds (skip top third)
    middle_slice = slice(height // 3, 2 * height // 3)
    # middle_slice = slice(0, height // 3)
    lower_slice = slice(2 * height // 3, height)
    edge_count_middle = np.count_nonzero(edge_mask[middle_slice, :])
    edge_count_lower = np.count_nonzero(edge_mask[lower_slice, :])
    ratio_param = 0.001  # Prevent division by zero.
    edge_ratio = edge_count_lower / (edge_count_middle + ratio_param)
    print(f"{os.path.basename(image_path)} - Edge Ratio: {edge_ratio:.2f}, Middle: {edge_count_middle}, Lower: {edge_count_lower}")

    # Create an output image that will show the edges.
    # Start with a black image and draw edges in red.
    annotated_img = np.zeros((height, width, 3), dtype=np.uint8)
    # Set the edge pixels to red (BGR: 0, 0, 255)
    annotated_img[edge_mask] = (0, 0, 180)

    # Annotate the image based on the computed edge ratio.
    if edge_ratio >= ratio_threshold:
        label = "CARPET"
        color = (0, 255, 0)  # green
        cv.putText(annotated_img, f"{label} ({edge_ratio:.2f})", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv.LINE_AA)
        # Draw a bold rectangle border around the image
        cv.rectangle(annotated_img, (10, 10), (width - 10, height - 10), color, 5)
    else:
        label = "OBSTACLE"
        color = (0, 0, 255)  # Green
        cv.putText(annotated_img, f"{label} ({edge_ratio:.2f})", (50, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv.LINE_AA)
    
    # Draw the separation lines
    color = (255, 0, 0)
    thickness = 2
    cv.line(annotated_img, (0, height // 3), (width, height // 3), color, thickness)
    cv.line(annotated_img, (0, 2 * height // 3), (width, 2 * height // 3), color, thickness)


    # Write the image
    cv.imwrite(output_path, annotated_img)
    print(f"Saved: {output_path}")

def oldmain():
    # Get the directory where this script is located.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_folder = "/home/sblackmore/Documents/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt"
    output_folder = os.path.join(script_dir, "edge_images")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each JPG image in the folder.
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, f"edges_{image_file}")
        process_image(image_path, output_path)

def main():
    # Get the image
    # You need to specify the image manually!
    img_path = '/home/sblackmore/Documents/paparazzi/prototyping/collected_datasets/Test3_7maart_tapijt/1226448613.jpg'
    output_path = '/home/sblackmore/Documents/paparazzi/test_folder/test_images/single_images/1226448613.jpg'

    process_image(img_path, output_path)

if __name__ == "__main__":
    main()
‘’’