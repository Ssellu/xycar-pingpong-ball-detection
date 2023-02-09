import torch
from torch.utils.data import Dataset
import os,sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from imgaug import augmenters as iaa
import cv2
import time
import random

from util.tools import get_transformations

class Yolodata(Dataset):
    file_dir = ""
    anno_dir = ""
    file_txt = ""
    train_dir = "./data/train"
    train_txt = "all.txt"
    valid_dir = "./data/val"
    valid_txt = "all.txt"
    class_str = ['left', 'right', 'stop', 'crosswalk', 'uturn', 'traffic_light']
    num_class = None
    img_data = []
    def __init__(self, is_train=True, transform=None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param['class']
        self.total_data = []
        self.cfg_param = cfg_param

        if self.is_train:
            self.file_dir = self.train_dir+"/JPEGImages/"
            self.file_txt = self.train_dir+"/ImageSets/"+self.train_txt
            self.anno_dir = self.train_dir+"/Annotations/"
        else:
            self.file_dir = self.valid_dir+"/JPEGImages/"
            self.file_txt = self.valid_dir+"/ImageSets/"+self.valid_txt
            self.anno_dir = self.valid_dir+"/Annotations/"

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding='UTF-8', errors='ignore') as f:
            img_names = [ i.replace("\n", "") for i in f.readlines()]
        for i in img_names:
            if os.path.exists(self.file_dir + i + ".jpg"):
                img_data.append(i+".jpg")
            elif os.path.exists(self.file_dir + i + ".JPG"):
                img_data.append(i+".JPG")
            elif os.path.exists(self.file_dir + i + ".png"):
                img_data.append(i+".png")
            elif os.path.exists(self.file_dir + i + ".PNG"):
                img_data.append(i+".PNG")
            else:
                continue
            orig_img, orig_bbox = self._get_origin_img_bbox(filename=img_data[-1])
            print(f'orig_bbox : {orig_bbox}')
            self.total_data = [(orig_img, orig_bbox)]
            if is_train:
                self._append_aug_img(orig_img, orig_bbox)
            # self._append_aug_img(orig_img, orig_bbox)
        random.shuffle(self.total_data)


    def _get_origin_img_bbox(self, filename:str):
        img_path = self.file_dir + filename
        with open(img_path, 'rb') as f:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)


        #if anno_dir is didnt exist, Test dataset
        if not os.path.isdir(self.anno_dir):
            bbox = np.array([[0,0,0,0,0]], dtype=np.float64)
            img, _ = self.transform((img, bbox))
            return img, None

        txt_name = filename
        for ext in ['.png','.PNG','.jpg','.JPG']:
            txt_name = txt_name.replace(ext, ".txt")
        anno_path = self.anno_dir + txt_name

        #skip if no anno_file
        if not os.path.exists(anno_path):
            return

        bbox = []
        with open(anno_path, 'r') as f:
            for line in f.readlines():
                line = line.replace("\n","")
                gt_data = [ l for l in line.split(" ")]
                #skip when no data
                if len(gt_data) < 5:
                    continue
                cx, cy, w, h = float(gt_data[1]), float(gt_data[2]), float(gt_data[3]), float(gt_data[4])
                bbox.append([float(gt_data[0]), cx, cy, w, h])

        #Change gt_box type
        bbox = np.array(bbox)
        print('==' * 30)
        print(bbox)
        print('==' * 30)
        #skip empty target
        if bbox.shape[0] == 0:
            return

        return img, bbox

    def _append_aug_img(self, orig_img, orig_bbox):
        #data augmentation
        tranforms = self._get_bunch_transforms()
        # tranforms = [self.transform]

        for transform in tranforms:
            print(f'!!!origgg : {orig_bbox}')
            img, bbox = transform((orig_img, torch.tensor(orig_bbox)))

            #############################
            # Image Load
            image = img.permute(1, 2, 0).numpy().copy() #.astype(np.int8)
            image_width = image.shape[1]
            image_height = image.shape[0]

            print(f'image_width :{image_width}')
            print(f'image_height :{image_height}')
            print(f'bbox : {bbox}')
            for norms in bbox:
                x = norms[1]
                y = norms[2]
                w = norms[3]
                h = norms[4]

                # Draw Rectangles
                print(f'shape : {image.shape}')
                image = cv2.rectangle(image, (int((x - w/2) * image_width), int((y - h/2) * image_height)), (int((x + w/2) * image_width), int((y + h/2) * image_height)),
                                   (random.randrange(256), random.randrange(256), random.randrange(256)), 1)
            cv2.imshow('test', image)
            cv2.waitKey(0)

            #############################
            if bbox.shape[0] != 0:
                batch_idx = torch.zeros(bbox.shape[0])
                #batch_idx, cls, x, y, w, h
                target_data = torch.cat((batch_idx.view(-1,1),bbox),dim=1)
                self.total_data.append((img, target_data))


    def _get_bunch_transforms(self):
        cfg_param = self.cfg_param
        transforms = [transform for transform in
                    [get_transformations(cfg_param=cfg_param, is_train=True)]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.Add, value=value) for value in [25, 45, -25, 45]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.AdditiveGaussianNoise, scale=scale*255) for scale in [0.03, 0.05, 0.10, 0.20]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.SaltAndPepper, p=p) for p in [0.01, 0.02, 0.03]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.Cartoon)]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.GaussianBlur, sigma=sigma) for sigma in [0.25, 0.5, 1.0]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.MotionBlur, seed=seed) for seed in [0, 72, 144, 216, 288]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.ChangeColorTemperature, kelvin=kelvin) for kelvin in [8000, 16000]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.RemoveSaturation, mul=mul) for mul in [0.25]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.GammaContrast, gamma=gamma) for gamma in [0.50, 0.81, 1.12, 1.44]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.SigmoidContrast, gain=gain) for gain in [5.1, 17.1, 14.4]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.HistogramEqualization, to_colorspace=to_colorspace) for to_colorspace in [iaa.HistogramEqualization.HSV,iaa.HistogramEqualization.HLS,iaa.HistogramEqualization.Lab]]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.Sharpen, alpha=alpha, lightness=lightness) for alpha,lightness in zip([1, 1, 1, 1], [1.5, 1.2, 0.5, 0.8])]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.Emboss, alpha=alpha, strength=strength) for alpha,strength in zip([1, 1, 1], [0.2, 0.3, 0.4])]
                    + [get_transformations(cfg_param=cfg_param, is_train=True, augmenter=iaa.CropAndPad, px=px) for px in [(-2,0,0,0), (0,2,0,-2), (2,0,0,0), (0,-2,0,2)]]
        ]
        print(f'trans : {transforms}')
        return transforms

    def __getitem__(self, index):
        return self.total_data[index]


    def __len__(self):
        return len(self.total_data)

