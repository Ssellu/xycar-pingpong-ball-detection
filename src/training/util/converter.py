import torch
import numpy as np


def minmax2cxcy(box):
    if len(box) != 4:
        return torch.FloatTensor([0,0,0,0])
    else:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]

        if cx - w/2 < 0 or cx + w/2 > 1:
            w -= 0.001
        if cy - h/2 < 0 or cy + h/2 > 1:
            h -= 0.001
        box[0] = cx
        box[1] = cy
        box[2] = w
        box[3] = h

def cxcy2minmax(box):
    y = box.new(box.shape)
    xmin = box[...,0] - box[...,2] / 2
    ymin = box[...,1] - box[...,3] / 2
    xmax = box[...,0] + box[...,2] / 2
    ymax = box[...,1] + box[...,3] / 2
    y[...,0] = xmin
    y[...,1] = ymin
    y[...,2] = xmax
    y[...,3] = ymax
    return y

def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y