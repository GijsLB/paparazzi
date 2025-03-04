import cv2
import numpy as np
import glob
from random import randrange
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
import os

# ======= STEP 1: Load Training Images & Masks =======
images = glob.glob('/home/gijs/Desktop/MAV/notebooks_MAV/MAV_ML_notebook/data/*c.jpg')
labels = glob.glob('/home/gijs/Desktop/MAV/notebooks_MAV/MAV_ML_notebook/data/*m.jpg')

print(f"Found {len(images)} images:", images)
print(f"Found {len(labels)} masks:", labels)

# ======= STEP 2: Convert Images to Training Data =======
X_vec = []
y_vec = []
maxfiles = 500  # Limit the number of images used for training
samples_per_image = 75000  # Number of pixels sampled per image

for f in images:
    lf = f.replace('c.jpg', 'm.jpg')  # Find the corresponding mask
    if lf in labels:
        maxfiles -= 1
        if maxfiles <= 0:
            break

        img = cv2.imread(f)
        msk = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)  # Load mask in grayscale
        h, w, d = img.shape

        print(f"Processing: {f}, Mask: {lf}, Size: {w}x{h}")

        # Convert to YUV color space
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        for i in range(samples_per_image):
            x = randrange(2, w-3)
            y = randrange(4, h-2)

            # Get pixel color values in YUV
            p = yuv[y, x]
            # Get corresponding mask label (0 = background, 255 = object)
            m = int(msk[y, x])
            m = 0 if m < 127 else 255

            # Store training data
            X_vec.append([int(p[0]), int(p[1]), int(p[2])])
            y_vec.append([m])

print(f'Dataset size: {len(X_vec)} samples')

# ======= STEP 3: Split Training and Test Data =======
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y_vec, test_size=0.2, stratify=y_vec, random_state=1
)

print(f'Train set: {len(X_train)}, Test set: {len(X_test)}')

# ======= STEP 4: Train the Decision Tree Classifier =======
dt = DecisionTreeClassifier(max_depth=2, random_state=0)
dt.fit(X_train, y_train)

# Evaluate accuracy
y_pred = dt.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {round(score, 3)}')

# Print the decision tree rules
print(export_text(dt, feature_names=['Y', 'U', 'V']))

# ======= STEP 5: Apply the Trained Model to a Custom Image =======
image_path = "/home/gijs/paparazzi/prototyping/cyberzoo_poles/20190121-135009/81578080.jpg"
img = cv2.imread(image_path)
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

if img is None:
    print("Error: Image not found or path incorrect.")
    exit()

h, w, d = img.shape

# Convert to YUV
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# Reshape image into feature vector
X_run = yuv.reshape(int(h * w), int(d))

# Predict using the trained classifier
y_pred = dt.predict(X_run)

# Reshape prediction back into an image mask
msk = y_pred.reshape(h, w)

# Overlay mask on the original image (change U-channel)
classified_img = img.copy()
classified_img[:, :, 1] = msk[:, :]

# ======= Display & Save Both Images =======
# Resize images for easier labeling (e.g., 50% of original size)
scale_factor = 0.5  # Adjust this to make the images smaller/larger
resized_original = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
resized_classified = cv2.resize(classified_img, (int(w * scale_factor), int(h * scale_factor)))

# Save images
output_folder = "/home/gijs/Desktop/MAV/notebooks_MAV/MAV_ML_notebook/results/"
os.makedirs(output_folder, exist_ok=True)

cv2.imwrite(os.path.join(output_folder, "original_image.jpg"), img)
cv2.imwrite(os.path.join(output_folder, "classified_image.jpg"), classified_img)

# Display images
fig, ax = plt.subplots(1, 2, figsize=(8, 4))  # Adjust figure size
ax[0].imshow(cv2.cvtColor(resized_original, cv2.COLOR_BGR2RGB))
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(cv2.cvtColor(resized_classified, cv2.COLOR_YUV2RGB))  # Convert back to RGB for visualization
ax[1].set_title("Classified Image")
ax[1].axis("off")

plt.show()
