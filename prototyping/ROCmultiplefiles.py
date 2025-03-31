import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append("/home/berg/paparazzi/prototyping/decision_logic")

import sys
sys.path.append("/home/berg/paparazzi/prototyping/decision_logic")
from no_paspoes_adjustedforROC import process_image

sys.path.append("/home/berg/paparazzi/prototyping/decision_logic")
from ROCofground import get_ROC_curve


input_dir = os.path.expanduser("~/paparazzi/prototyping/Groundtruth/testlabel1")
GT_dir = os.path.expanduser("~/paparazzi/prototyping/Groundtruth/labbeledGT")

save_path = '/home/berg/paparazzi/prototyping/Groundtruth/ROC_curves/'
save_graphname = 'ROCdownsizing3'



# threshold_values = np.arange(0, 210, 10).tolist()
# threshold_values = np.arange(0, 1.1, 0.1).tolist()
# threshold_values = np.arange(1, 201, 10).tolist()
threshold_values = np.arange(1, 21, 1).tolist()

TPR = []
FPR = []
i = 1

for file_path in glob.glob(os.path.join(input_dir, "*.jpg")):
    print(i)
    i += 1

    file_name = os.path.basename(file_path).replace(".jpg", "m.png")  # Replace .jpg with .m.png
    gt_file_path = os.path.join(GT_dir, file_name)  # Full path in GT_dir

     
    

    if os.path.exists(gt_file_path):

        GT = cv2.imread(gt_file_path, cv2.IMREAD_GRAYSCALE)

        _, GTbinairy = cv2.threshold(GT, 127, 1, cv2.THRESH_BINARY)

        tp, fp = get_ROC_curve(file_path, GTbinairy, process_image, threshold_values)

        TPR += [tp]
        FPR += [fp]
        
        
    else:
        print(f"No match found for: {gt_file_path}")

print(len(TPR), len(TPR[0]))

TPR_avg = np.mean(TPR, axis=0)
FPR_avg = np.mean(FPR, axis=0)



plt.figure(1)
for i in range(len(TPR)):
    plt.plot(FPR[i], TPR[i], label=f'Line {i+1}', marker='o')

plt.xlabel('TPR')
plt.ylabel('TPR')
plt.title('TPR vs. Threshold Values')
plt.legend(loc='best', fontsize='small', ncol=2)  # Adjust legend settings
plt.grid(True)
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()


plt.figure(2)
plt.plot([0,1], [0,1], '--' )
plt.plot(FPR_avg, TPR_avg, 'b', marker='.')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title("Average ROC curve of ground detection. Parameter: downsizing scale")
plt.savefig(os.path.join(save_path, save_graphname))


plt.figure(3)
plt.plot(threshold_values, TPR_avg, 'b', label = "TPR", marker='o')
plt.plot(threshold_values, FPR_avg, 'r', label = "FPR",  marker='o')
plt.xlabel('threshold')
plt.ylabel('FP')
plt.legend()
#plt.xlim([0,1])
plt.ylim([0,1])
plt.show()
