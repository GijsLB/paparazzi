"""
    filters.py
    In this document there is experimentation with multiple filters.
    Focus is laid on gradients and edge detection
"""

import cv2 as cv
import matplotlib.pyplot as plt
import os

# print(os.getcwd())

img = cv.imread('/home/sblackmore/Documents/paparazzi/prototyping/AE4317_2019_datasets/cyberzoo_poles/20190121-135009/81011442.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "No file"

img_rotated = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)

laplacian = cv.Laplacian(img_rotated, cv.CV_64F)
sobely = cv.Sobel(img_rotated, cv.CV_64F, 1, 0, ksize=5)
sobelx = cv.Sobel(img_rotated, cv.CV_64F, 0, 1, ksize=5)

plt.subplot(2, 2, 1)
plt.imshow(img_rotated, cmap= 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(laplacian, cmap= 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3)
plt.imshow(sobely, cmap= 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(sobelx, cmap= 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.show()
