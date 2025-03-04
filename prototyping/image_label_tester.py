import cv2
import numpy as np
import glob
import os

# === CONFIGURATION ===
image_folder = "/home/gijs/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles/20190121-135009"
output_folder = "/home/gijs/paparazzi/prototyping/our_labelled_data"
brush_size = 10  # Brush size for visualization

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Get all images that need labeling (skip already labeled ones)
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
image_paths = [p for p in image_paths if not os.path.exists(p.replace(".jpg", "m.png"))]  # Change mask extension to PNG

# Drawing settings
points = []  # List of points for polygon selection
mask = None
display_image = None
original_image = None  # Keep track of the unmodified image


def draw_polygon(image, points):
    """Draw polygon lines based on clicked points."""
    for i in range(len(points) - 1):
        cv2.line(image, points[i], points[i + 1], (0, 255, 0), 2)
    if len(points) > 2:  # Close polygon visually
        cv2.line(image, points[-1], points[0], (0, 255, 0), 2)


def on_mouse(event, x, y, flags, param):
    """Mouse callback to select points."""
    global points, display_image
    if event == cv2.EVENT_LBUTTONDOWN:  # Left click: add point
        points.append((x, y))
        draw_polygon(display_image, points)
        cv2.imshow("Labeling Tool", display_image)


def fill_polygon():
    """Fill the selected polygon area in the mask and display."""
    global mask, points, display_image

    if len(points) < 3:  # Needs at least 3 points to form an area
        print("Not enough points to form an enclosed area!")
        return

    # Create a filled mask using the polygon
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Overlay mask on display image (make it visible)
    overlay = display_image.copy()
    overlay[mask > 0] = (0, 255, 0)  # Green overlay for labeled area
    cv2.addWeighted(overlay, 0.4, display_image, 0.6, 0, display_image)

    cv2.imshow("Labeling Tool", display_image)

    # Reset points so a new polygon can be started
    points = []


def process_images():
    """Main function to process images for labeling."""
    global mask, display_image, original_image, points

    for image_path in image_paths:
        print(f"Labeling: {image_path}")

        # Load and prepare the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading: {image_path}")
            continue

        # Rotate & maintain original size
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = image.shape[:2]

        # Resize image **larger** (150%) for easier labeling
        scale = 1.5  
        resized_w = int(w * scale)
        resized_h = int(h * scale)
        image = cv2.resize(image, (resized_w, resized_h))

        # Create an empty mask
        mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
        display_image = image.copy()
        original_image = image.copy()  # Store original for saving

        # Show the image
        cv2.imshow("Labeling Tool", display_image)
        cv2.setMouseCallback("Labeling Tool", on_mouse)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("f"):  # Fill enclosed polygon
                fill_polygon()

            elif key == ord("s"):  # Save and move to next
                label_path = os.path.join(output_folder, os.path.basename(image_path).replace(".jpg", "m.png"))
                original_path = os.path.join(output_folder, os.path.basename(image_path))

                # Save both the labeled mask as **PNG** and the original image
                cv2.imwrite(label_path, mask)
                cv2.imwrite(original_path, original_image)
                print(f"Saved: {label_path} & {original_path}")
                break

            elif key == ord("z"):  # Undo last draw
                points = []
                mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
                display_image = original_image.copy()
                cv2.imshow("Labeling Tool", display_image)

        cv2.destroyAllWindows()

    print("✅ Labeling completed for all images!")


# Run the labeling process
process_images()
