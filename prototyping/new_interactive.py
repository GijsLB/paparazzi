import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import matplotlib
matplotlib.use('TkAgg')  # GUI backend

# Load raw UYVY image (as if read from drone camera)
image_path = os.path.expanduser("~/paparazzi/prototyping/AE4317_2019_datasets/sim_poles/20190121-160844/34836000.jpg")
raw = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
if raw is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Simulate UYVY format
height, width, _ = raw.shape
uyvy = np.empty((height, width, 3), dtype=np.uint8)

for y in range(height):
    for x in range(0, width, 2):
        b1, g1, r1 = raw[y, x]
        b2, g2, r2 = raw[y, x+1]

        # Convert RGB to YUV for 2 pixels
        y1 = 0.299*r1 + 0.587*g1 + 0.114*b1
        u  = -0.169*r1 - 0.331*g1 + 0.5*b1 + 128
        y2 = 0.299*r2 + 0.587*g2 + 0.114*b2
        v  = 0.5*r1 - 0.419*g1 - 0.081*b1 + 128

        uyvy[y, x] = [u, y1, v]
        uyvy[y, x+1] = [u, y2, v]

im = uyvy.astype(np.uint8)

# Initial filter ranges
y_low, y_high = 75, 250
u_low, u_high = 110, 155
v_low, v_high = 50, 145

def update(val):
    y_low, y_high = s_y_low.val, s_y_high.val
    u_low, u_high = s_u_low.val, s_u_high.val
    v_low, v_high = s_v_low.val, s_v_high.val

    mask = (im[:,:,1] >= y_low) & (im[:,:,1] <= y_high) & \
           (im[:,:,0] >= u_low) & (im[:,:,0] <= u_high) & \
           (im[:,:,2] >= v_low) & (im[:,:,2] <= v_high)

    filtered = np.zeros_like(im[:,:,1])
    filtered[mask] = 255

    ax_filtered.imshow(cv2.rotate(filtered, cv2.ROTATE_90_COUNTERCLOCKWISE), cmap='gray')
    fig.canvas.draw_idle()

# Display setup
fig, (ax_original, ax_filtered) = plt.subplots(1, 2, figsize=(10, 5))
ax_original.imshow(cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE))
ax_original.set_title("Original Image")
ax_filtered.imshow(np.zeros_like(im[:,:,1]), cmap='gray')
ax_filtered.set_title("Filtered Image")

# Slider axes
axcolor = 'lightgoldenrodyellow'
ax_y_low = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor=axcolor)
ax_y_high = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor=axcolor)
ax_u_low = plt.axes([0.2, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_u_high = plt.axes([0.2, 0.14, 0.65, 0.03], facecolor=axcolor)
ax_v_low = plt.axes([0.2, 0.18, 0.65, 0.03], facecolor=axcolor)
ax_v_high = plt.axes([0.2, 0.22, 0.65, 0.03], facecolor=axcolor)

# Sliders
s_y_low = Slider(ax_y_low, 'Y Low', 0, 255, valinit=y_low)
s_y_high = Slider(ax_y_high, 'Y High', 0, 255, valinit=y_high)
s_u_low = Slider(ax_u_low, 'U Low', 0, 255, valinit=u_low)
s_u_high = Slider(ax_u_high, 'U High', 0, 255, valinit=u_high)
s_v_low = Slider(ax_v_low, 'V Low', 0, 255, valinit=v_low)
s_v_high = Slider(ax_v_high, 'V High', 0, 255, valinit=v_high)

# Hook up callbacks
s_y_low.on_changed(update)
s_y_high.on_changed(update)
s_u_low.on_changed(update)
s_u_high.on_changed(update)
s_v_low.on_changed(update)
s_v_high.on_changed(update)

plt.show()