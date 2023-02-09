import json
import cv2
from cv2 import undistort
import matplotlib.pyplot as plt
import os
import numpy as np

anno_file_path = os.path.join("../training/data/train/Annotations", "002.txt")
image_file_path = os.path.join("../training/data/train/JPEGImages", "002.jpg")
json_file_path = os.path.join(".", "000076.json")  # TODO - Calibration Information

window_name = "Perception"



with open(json_file_path, "r") as json_file:
    labeling_info = json.load(json_file)

with open(anno_file_path, "r") as f:
    labels =[[float(s.replace('\n', '')) for s in ff.split(' ')] for ff in f.readlines()]
    labels = np.array(labels, dtype=np.float32)
    labels = labels[:, 1:]

image = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)

camera_matrix = np.asarray(labeling_info["calib"]["cam01"]["cam_intrinsic"], dtype=np.float32)
dist_coeff = np.asarray(labeling_info["calib"]["cam01"]["distortion"], dtype=np.float32)
undist_image = cv2.undistort(image, camera_matrix, dist_coeff, None, None)

class_names = 'ball'
boxes_2d = labels

CAMERA_HEIGHT = 1.3

# distance = f * height / img(y)
# 종/횡 방향으로 분리된 거리가 아닌, 직선거리
# FOV 정보를 알면 -> 종/횡 분리가 가능하다.

index = 0
for class_name, bbox in zip(class_names, boxes_2d):
    xmin, ymin, xmax, ymax = bbox
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
        continue


    width = xmax - xmin
    height = ymax - ymin

    # Normalized Image Plane
    y_norm = (ymax - camera_matrix[1][2]) / camera_matrix[1][1]

    distance = 1 * CAMERA_HEIGHT / y_norm

    print(int(distance))
    cv2.rectangle(undist_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
    cv2.putText(undist_image, f"{index}-{class_name}-{int(distance)}", (xmin, ymin+25), 1, 2, (255, 255, 0), 2)
    index += 1

display_image = cv2.cvtColor(undist_image, cv2.COLOR_BGR2RGB)
plt.imshow(display_image)
plt.show()