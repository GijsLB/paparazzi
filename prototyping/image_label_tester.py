import cv2
import numpy as np
import glob
import os

# === CONFIGURATION ===
image_folder = "/home/gijs/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles/20190121-135009"
brush_size = 10  # Brush size for drawing

# Get all images that need labeling
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))  # Load all JPG files
image_paths = [p for p in image_paths if not p.endswith("m.jpg")]  # Exclude already labeled images

# Brush settings
drawing = False
label_type = 1  # 1 = Obstacle, 0 = Ground

def draw(event, x, y, flags, param):
    global drawing, mask, label_type
    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:  # Draw while moving
        cv2.circle(mask, (x, y), brush_size, label_type * 255, -1)
        cv2.circle(display_image, (x, y), brush_size, (0, 255, 0) if label_type == 0 else (0, 0, 255), -1)
        cv2.imshow("Labeling Tool", display_image)
    elif event == cv2.EVENT_LBUTTONUP:  # Stop drawing
        drawing = False

# Loop through images
for image_path in image_paths:
    print(f"Labeling: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))  
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Empty mask
    display_image = image.copy()

    # Open window
    cv2.imshow("Labeling Tool", display_image)
    cv2.setMouseCallback("Labeling Tool", draw)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Save & move to next
            label_path = image_path.replace(".jpg", "m.jpg")  # Save as *_m.jpg
            cv2.imwrite(label_path, mask)
            print(f"Saved: {label_path}")
            break
        elif key == ord("1"):  # Set to obstacle
            label_type = 1
            print("Label Mode: Obstacle (Red)")
        elif key == ord("0"):  # Set to ground
            label_type = 0
            print("Label Mode: Ground (Green)")
        elif key == ord("z"):  # Undo last draw (reset mask)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            display_image = image.copy()
            cv2.imshow("Labeling Tool", display_image)
            print("Undo last changes!")

    cv2.destroyAllWindows()

print("✅ Labeling completed for all images!")
