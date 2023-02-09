from dataloader.yolodata import Yolodata
from dataloader.data_transforms import get_transformations
if __name__ == '__main__':
    cfg_param = {'in_height':352, 'in_width':352, 'class':1}
    eval_transforms = get_transformations(cfg_param, is_train = False)
    train_data = Yolodata(is_train = True,
                         cfg_param = cfg_param)
    print(train_data)
    print(train_data.__len__())
    for data in train_data:
        pass
