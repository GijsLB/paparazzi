import cv2
import numpy as np
import os

GT = cv2.imread("/home/berg/paparazzi/prototyping/Groundtruth/labbeledGT/740752450m.png").flatten
unfiltered = "/home/berg/paparazzi/prototyping/Groundtruth/labbeledGT/740752450.jpg" 
threshold_values = np.arange(0, 1.1, 0.1).tolist()
print(threshold_values)


def get_ROC_curve(image, ground_truth, Filter, thresholds):
 
    totalP = np.sum(ground_truth == 255)  # Count positive pixels (white)
    totalN = np.sum(ground_truth == 0)    # Count negative pixels (black)
    
    TPR = []  # True Positive Rate
    FPR = []  # False Positive Rate
    
    for threshold in thresholds:
        filtered_image = Filter(image, threshold).flatten # Apply filter
        
        TP = np.sum((filtered_image == 255) & (ground_truth == 255))  # True Positives
        FP = np.sum((filtered_image == 255) & (ground_truth == 0))    # False Positives
        
        TPR.append(TP / totalP if totalP > 0 else 0)
        FPR.append(FP / totalN if totalN > 0 else 0)
    
    return TPR, FPR

get_ROC_curve(unfiltered, GT, Groundfilter, threshold_values)

plt.figure()
plt.plot(FPR, TPR, 'b')
plt.xlabel('TP')
plt.ylabel('FP')

#Groundfilter sould be a function
#inputs: unfiltered image, scalar value of config vairable to be evaluated
#output: black and white filtered image