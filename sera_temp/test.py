import glob
import cv2

import random


def show_bbox(directory: str):
    image_path_list = glob.glob(f"{directory}/*.*p*")
    annotation_path_list = glob.glob(f"{directory}/*.txt")
    for image_path, annotation in zip(image_path_list, annotation_path_list):
        print(f'image : {image_path} / anno : {annotation}')

        # Image Load
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_width = image.shape[1]
        image_height = image.shape[0]
        # Annotation Load
        # Format : <object-class> <x center> <y center> <width> <height>
        with open(annotation, 'r') as f:
            anno_lst = f.readlines()

        for anno in anno_lst:
            norms = [float(n) for n in anno.split(' ')]
            x = norms[1]
            y = norms[2]
            w = norms[3]
            h = norms[4]

            # Draw Rectangles
            image = cv2.rectangle(image, (int((x - w/2) * image_width), int((y - h/2) * image_height)), (int((x + w/2) * image_width), int((y + h/2) * image_height)),
                                  (random.randrange(256), random.randrange(256), random.randrange(256)), 1)
        cv2.imshow('test', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    show_bbox('images')
