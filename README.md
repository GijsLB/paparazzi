# AE4317 Autonomous Flight of Micro Air Vehicles (2024/25 Q3)  
### Group 1 – Final Submission

This repository is the result of the **AE4317 Autonomous Flight of Micro Air Vehicles** course, where **Group 1** successfully designed an autonomous MAV capable of flying through an obstacle course.

Throughout the course, we explored multiple perception strategies, including:

- **Edge Detection**
- **Optical Flow**
- **Horizon Detection using YUV Filtering**

The final competition-day implementation used **YUV-based horizon detection**, though each strategy contributed to our learning and development. While the final code doesn't reflect all our efforts, this `README` highlights key files and folders that document our progress and experiments.

---

## 🔬 Prototyping Efforts

Most of our experimentation and development happened in: `~/paparazzi/prototyping/`

This folder contains various Python scripts showcasing our perception pipeline development.

### Noteworthy Files:

- `new_interactive.py`  
  Interactive YUV filter tool for fine-tuning thresholds on any image.

- `image_label_tester.py`  
  Tool to generate labeled datasets for training classifiers (used during our decision tree experiments).

- `decision_logic/no_paspoes.py`  
  Our final **Python-based perception logic**. The corresponding C implementation contains important adaptations.

- `ROCmultiplefiles.py`
- `ROCofground.py`
- `no_paspoes_adjustedforROC.py`
  Scripts used to make roc curves of the ground filter.
---

## 🌅 Horizon Detection using YUV Filter

Located in: `~/paparazzi/sw/airborne/modules/computer_vision/`


### Key Files:

- `cv_detect_color_object.c`  
  Core of the YUV color filtering algorithm.

- `orange_avoider.c`  
  Steering logic based on filtered regions (adapted from the standard module).

### Supporting Config Files:

- `~/paparazzi/conf/flight_plans/tudelft/course_orangeavoid_cyberzoo.xml`
- `~/paparazzi/conf/bebop_course_orangeavoid.xml`

> ⚠️ We **did not** create a new module, but chose to **adapt the existing Orange Avoider module** to suit our final perception logic.

---

## 🧪 Other Strategies (Explored but Not Used in Final)

### Edge Detection  
*Implemented by:* **Sean**

### Optical Flow  
*Implemented by:* **Harm & Jasper**

---

We hope this README gives proper credit to the effort that went into the various ideas and experiments throughout the course.
