import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


import sys
sys.path.append("/home/berg/paparazzi/prototyping/decision_logic")
from no_paspoes_adjustedforROC import process_image

# ======================
# CONFIGURABLE VARIABLES
# ======================
SCALE_F_X = 5
SCALE_F_Y = 5
## use these in /home/berg/paparazzi/prototyping/decision_logic/no_paspoes_adjustedforROC.py
NUM_COLUMNS = 100
NUM_BLOCKS_PER_COLUMN = 40
# Suppose you want the threshold line at 50% image height
HEIGHT_TRANSITION_FRACTION = 0.3
# “Lenient” vs. “Strict” sets
# lenient => we use small X_WHITE (so fewer whites required) & large Y_BLACK
X_WHITE_LENIENT = 1
Y_BLACK_LENIENT = 5
# strict => bigger X_WHITE, smaller Y_BLACK
X_WHITE_STRICT = 1
Y_BLACK_STRICT = 5

THRESH_OBSTACLE = 0.4


Ymin = 90
Ymax = 255
Umin = 75
Umax = 115
Vmin = 69
Vmax = 145


##

# image_name = "780418796.jpg"
# input_dir = os.path.expanduser("~/paparazzi/prototyping/Groundtruth/testlabel1")
# image_path = os.path.join(input_dir, image_name)


# GT_name = "780418796m.png"
# GT_dir = os.path.expanduser("~/paparazzi/prototyping/Groundtruth/labbeledGT")
# GT_path = os.path.join(GT_dir, GT_name)

# GT = cv2.imread(GT_path, cv2.IMREAD_GRAYSCALE)

# _, GTbinairy = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)

# if GT_path is None:
#         raise FileNotFoundError(f"Error: Could not load image at {GT_path}")

# threshold_values = np.arange(1, 11, 1).tolist()


# threshold_values = np.arange(0, 210, 10).tolist()
#threshold_values = np.arange(0, 1.1, 0.1).tolist()
#threshold_values = np.arange(1, 201, 10).tolist()

def get_ROC_curve(image, ground_truth, FilterFunc, thresholds):

    totalP = np.sum(ground_truth == 1)  # Count positive pixels (white)
    totalN = np.sum(ground_truth == 0)    # Count negative pixels (black)
    
    TPR = []  # True Positive Rate
    FPR = []  # False Positive Rate

    GTh, GTw = ground_truth.shape
    
    for i in thresholds:
    
        ##set configuration variable equal to i
        SCALE_F_X = i
        SCALE_F_Y = i
        #print(i)
       

        filtered_image = FilterFunc(SCALE_F_X, SCALE_F_Y, HEIGHT_TRANSITION_FRACTION,
                                   X_WHITE_LENIENT, Y_BLACK_LENIENT, X_WHITE_STRICT, Y_BLACK_STRICT,
                                   THRESH_OBSTACLE, image, GTh, GTw, Ymin)

        # plt.figure()
        # plt.imshow(filtered_image, cmap='gray')
        # plt.show()

        # plt.figure()
        # plt.imshow(ground_truth, cmap='gray')
        # plt.show()

        TP = np.sum((filtered_image == 1) & (ground_truth == 1))  # True Positives
        FP = np.sum((filtered_image == 1) & (ground_truth == 0))    # False Positives
        
        TPR.append(TP / totalP if totalP > 0 else 0)
        FPR.append(FP / totalN if totalN > 0 else 0)

    
    
        
    return TPR, FPR

    

# TPR, FPR = get_ROC_curve(image_path, GTbinairy, process_image, threshold_values)

# plt.figure()
# plt.plot([0,1], [0,1], '--' )
# plt.plot(FPR, TPR, 'b')
# plt.xlabel('TP')
# plt.ylabel('FP')
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.title("ROC curve of ground detection")
# plt.show()
# print(TPR)
# print(FPR)





# plt.figure()
# plt.plot(threshold_values, TPR, 'b', label = "TPR")
# plt.plot(threshold_values, FPR, 'r', label = "FPR")
# plt.xlabel('threshold')
# plt.ylabel('FP')
# plt.legend()
# #plt.xlim([0,1])
# #plt.ylim([0,1])
# plt.show()

# #Groundfilter should be a function
# #inputs: unfiltered image, scalar value of config vairable to be evaluated
# #output: black and white filtered image