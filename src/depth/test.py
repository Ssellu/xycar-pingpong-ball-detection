import cv2
from cv2 import undistort
import matplotlib.pyplot as plt
import os
import numpy as np
import glob


from common.util import *

# --* Globals *--
# YAML Properties
labeling_info = yaml_parser()
CAMERA_HEIGHT = labeling_info['xycar']['cameraHeight']
DISTANCE_WITH_ORIGIN = labeling_info['xycar']['distance']
camera_matrix = np.asarray(labeling_info['narrow_stereo']['cameraMatrix'], dtype=np.float32)
dist_coeff = np.asarray(labeling_info['narrow_stereo']['distortion'], dtype=np.float32)
image_path_list = sorted(glob.glob(f"../training/data/train/JPEGImages/left*.*"))
anno_path_list = sorted(glob.glob(f"../training/data/train/Annotations/left*.*"))
# ---------------



for image_file_path, anno_file_path in zip(image_path_list, anno_path_list):
    print('!!!', image_file_path, anno_file_path)
    with open(anno_file_path, "r") as f:
        labels = [[float(s.replace('\n', '')) for s in ff.split(' ')]
                for ff in f.readlines()]
        labels = np.array(labels, dtype=np.float32)
        labels = labels[:, 1:]
    image = cv2.imread(image_file_path, cv2.IMREAD_ANYCOLOR)
    undist_image = cv2.undistort(image, camera_matrix, dist_coeff, None, None)
    class_names = 'ball'
    boxes_2d = labels
    # distance = f * height / img(y)
    # 종/횡 방향으로 분리된 거리가 아닌, 직선거리
    # FOV 정보를 알면 -> 종/횡 분리가 가능하다.

    index = 0

    for bbox in boxes_2d:
        x, y, w, h = bbox
        xmin = int((x - w/2) * image.shape[1])
        ymin = int((y - h/2) * image.shape[0])
        xmax = int((x + w/2) * image.shape[1])
        ymax = int((y + h/2) * image.shape[0])
        if xmin < 0 or ymin < 0 or xmax < 0 or ymax < 0:
            continue

        width = xmax - xmin
        height = ymax - ymin

        # Normalized Image Plane
        y_norm = (ymax - camera_matrix[1][2]) / camera_matrix[1][1]
        distance = CAMERA_HEIGHT / y_norm - DISTANCE_WITH_ORIGIN

        print(f"{int(distance * 100)}cm")
        cv2.rectangle(undist_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
        cv2.putText(undist_image, f"{int(distance * 100)}cm",
                    (xmin, ymin+25), 1, 2, (255, 255, 0), 2)
        index += 1

    display_image = cv2.cvtColor(undist_image, cv2.COLOR_BGR2RGB)
    plt.imshow(display_image)
    plt.show()


