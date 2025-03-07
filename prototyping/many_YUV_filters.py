import cv2
import numpy as np
import os

# Define directory containing images
input_dir = "~/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles_panels/20190121-140205"
input_dir = os.path.expanduser(input_dir)

# Define filter thresholds
filters = {
    "orange": ((50, 200), (0, 120), (160, 220)),
    "black": ((55, 70), (0, 200), (0, 150)),
    "green": ((75, 250), (110, 155), (50, 145))
}

# Generate output directory name based on filter values
filter_desc = "_".join([f"{name}_{y[0]}-{y[1]}_{u[0]}-{u[1]}_{v[0]}-{v[1]}" for name, (y, u, v) in filters.items()])
output_dir = os.path.join(os.path.expanduser("~/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles_panels"), filter_desc)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define filter functions
def apply_filter(image, y_range, u_range, v_range):
    mask = (image[:,:,0] >= y_range[0]) & (image[:,:,0] <= y_range[1]) & \
           (image[:,:,1] >= u_range[0]) & (image[:,:,1] <= u_range[1]) & \
           (image[:,:,2] >= v_range[0]) & (image[:,:,2] <= v_range[1])
    
    filtered = np.zeros_like(image[:,:,0])
    filtered[mask] = 255  # White for detected areas
    return filtered

# Process each image in the directory
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith(".jpg"):
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)  # Convert to YUV
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Apply filters
        combined_result = [cv2.cvtColor(rotated_image, cv2.COLOR_YUV2BGR)]
        for name, (y_range, u_range, v_range) in filters.items():
            filtered = apply_filter(rotated_image, y_range, u_range, v_range)
            filtered_bgr = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            
            # Add filter name to image
            cv2.putText(filtered_bgr, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            combined_result.append(filtered_bgr)
        
        # Stack original + filters side by side
        output_image = np.hstack(combined_result)
        
        # Save output image
        output_path = os.path.join(output_dir, f"filtered_{filename}")
        cv2.imwrite(output_path, output_image)

print(f"Processed images saved in {output_dir}")
