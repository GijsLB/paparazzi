import cv2
import numpy as np
import os

# Define input directory containing images
input_dir = "~/paparazzi/prototyping/decision_logic/test_frames"
input_dir = os.path.expanduser(input_dir)

# Define green filter thresholds (YUV format)
green_filter = ((75, 250), (110, 155), (50, 145))

# Function to apply the green filter
def apply_green_filter(image, y_range, u_range, v_range):
    mask = (image[:, :, 0] >= y_range[0]) & (image[:, :, 0] <= y_range[1]) & \
           (image[:, :, 1] >= u_range[0]) & (image[:, :, 1] <= u_range[1]) & \
           (image[:, :, 2] >= v_range[0]) & (image[:, :, 2] <= v_range[1])
    
    filtered = np.zeros_like(image[:, :, 0])
    filtered[mask] = 255  # White for detected areas
    return filtered

# Process each image in the directory
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        
        # Convert to YUV color space
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        
        # Apply the green filter
        filtered_image = apply_green_filter(image_yuv, *green_filter)
        
        # Save the filtered image with '_greenfilter' appended
        output_path = os.path.join(input_dir, f"{filename.rsplit('.', 1)[0]}_greenfilter.jpg")
        cv2.imwrite(output_path, filtered_image)

print(f"Processed images saved in {input_dir}")
