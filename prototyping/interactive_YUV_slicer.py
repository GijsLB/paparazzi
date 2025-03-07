import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib
matplotlib.use('Qt5Agg')  # Use interactive backend



# Load the image
#image_path = "/home/gijs/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles/20190121-135009/81578080.jpg"
#image_path = "/home/gijs/Desktop/MAV/notebooks_MAV/YUV_notebook/DelFly_tulip.jpg"
#image_path = "/home/gijs/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/38615883.jpg"

import os
image_path = os.path.expanduser("~/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles_panels_mats/20190121-142935/38815902.jpg")

im = cv2.imread(image_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)  # Convert to YUV

# Default threshold values
y_low, y_high = 50, 200
u_low, u_high = 0, 120
v_low, v_high = 160, 220

# Function to apply filtering based on current slider values
def update(val):
    global y_low, y_high, u_low, u_high, v_low, v_high

    # Read slider values
    y_low, y_high = s_y_low.val, s_y_high.val
    u_low, u_high = s_u_low.val, s_u_high.val
    v_low, v_high = s_v_low.val, s_v_high.val

    # Apply filter
    mask = (im[:,:,0] >= y_low) & (im[:,:,0] <= y_high) & \
           (im[:,:,1] >= u_low) & (im[:,:,1] <= u_high) & \
           (im[:,:,2] >= v_low) & (im[:,:,2] <= v_high)
    
    filtered = np.zeros_like(im[:,:,0])
    filtered[mask] = 255  # Show detected areas in white

    # Update plot with rotated filtered image
    ax_filtered.imshow(cv2.rotate(filtered, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap='gray')
    fig.canvas.draw_idle()

# Create figure and axes
fig, (ax_original, ax_filtered) = plt.subplots(1, 2, figsize=(10, 5))

# Rotate the original image before displaying
ax_original.imshow(cv2.rotate(cv2.cvtColor(im, cv2.COLOR_YUV2RGB), cv2.ROTATE_90_COUNTERCLOCKWISE))
ax_original.set_title("Original Image")

ax_filtered.imshow(np.zeros_like(im[:,:,0]), cmap='gray')
ax_filtered.set_title("Filtered Image")

# Define slider positions
axcolor = 'lightgoldenrodyellow'
ax_y_low = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor=axcolor)
ax_y_high = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor=axcolor)
ax_u_low = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_u_high = plt.axes([0.2, 0.14, 0.65, 0.03], facecolor=axcolor)
ax_v_low = plt.axes([0.2, 0.18, 0.65, 0.03], facecolor=axcolor)
ax_v_high = plt.axes([0.2, 0.22, 0.65, 0.03], facecolor=axcolor)

# Create sliders
s_y_low = Slider(ax_y_low, 'Y Low', 0, 255, valinit=y_low)
s_y_high = Slider(ax_y_high, 'Y High', 0, 255, valinit=y_high)
s_u_low = Slider(ax_u_low, 'U Low', 0, 255, valinit=u_low)
s_u_high = Slider(ax_u_high, 'U High', 0, 255, valinit=u_high)
s_v_low = Slider(ax_v_low, 'V Low', 0, 255, valinit=v_low)
s_v_high = Slider(ax_v_high, 'V High', 0, 255, valinit=v_high)

# Attach update function to sliders
s_y_low.on_changed(update)
s_y_high.on_changed(update)
s_u_low.on_changed(update)
s_u_high.on_changed(update)
s_v_low.on_changed(update)
s_v_high.on_changed(update)

plt.show()
