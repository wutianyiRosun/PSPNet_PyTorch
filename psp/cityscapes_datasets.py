import torch
import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torchvision
import cv2
from torch.utils import data


class CityscapesDataSet(data.Dataset):
    """ 
       CityscapesDataSet is employed to load train set
       Args:
        root: the Cityscapes dataset path, 
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_train_list.txt, include partial path
        mean: bgr_mean (73.15835921, 82.90891754, 72.39239876)

    """
    def __init__(self, root, list_path, max_iters=None, crop_size=(512, 1024), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            #print(img_file)
            label_file = osp.join(self.root, name.split()[1])
            #print(label_file)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        print("length of dataset: ",len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.scale:
            f_scale = 0.5 + random.randint(0, 15) / 10.0  #random resize between 0.5 and 2 
            image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)
        
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        
        
        image = image.transpose((2, 0, 1)) # NHWC -> NCHW
        
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name

class CityscapesValDataSet(data.Dataset):
    """ 
       CityscapesDataSet is employed to load val set
       Args:
        root: the Cityscapes dataset path, 
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_val_list.txt, include partial path

    """
    def __init__(self, root, list_path, f_scale=0.5, mean=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.f_scale = f_scale
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for name in self.img_ids:
            img_file = osp.join(self.root, name.split()[0])
            #print(img_file)
            label_file = osp.join(self.root, name.split()[1])
            #print(label_file)
            image_name = name.strip().split()[0].strip().split('/',3)[3].split('.')[0]
            #print("image_name:  ",image_name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": image_name
            })

        print("length of dataset: ",len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        size = image.shape
        name = datafiles["name"]
        if self.f_scale !=1:
            image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation = cv2.INTER_LINEAR)
        #label = cv2.resize(label, None, fx=self.f_scale, fy=self.f_scale, interpolation = cv2.INTER_NEAREST)

        image = np.asarray(image, np.float32)

        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1)) # NHWC -> NCHW
        
        #print('image.shape:',image.shape)
        return image.copy(), label.copy(), np.array(size), name

class CityscapesTestDataSet(data.Dataset):
    """ 
       CityscapesDataSet is employed to load test set
       Args:
        root: the Cityscapes dataset path, 
         cityscapes
          ├── gtFine
          ├── leftImg8bit
        list_path: cityscapes_test_list.txt, include partial path

    """
    def __init__(self, root, list_path='test.txt', f_scale=1, mean=(128, 128, 128), ignore_label=255, set='test'):
        self.root = root
        self.list_path = list_path
        self.ignore_label = ignore_label
        self.mean = mean
        self.f_scale = f_scale
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        self.set = set
        for name in self.img_ids:
            #img_file = osp.join(self.root, "leftImg8bit/%s/%s" % (self.set, name))
            img_file = osp.join(self.root, name.strip())
            image_name = name.strip().split('/')[3].split('.')[0]
            self.files.append({
                "img": img_file,
                "name": image_name
            })
        print("lenth of dataset: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        #image = Image.open(datafiles["img"]).convert('RGB')
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        name = datafiles["name"]

        # resize
        if self.f_scale !=1:
            image = cv2.resize(image, None, fx=self.f_scale, fy=self.f_scale, interpolation = cv2.INTER_LINEAR)
        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name


if __name__ == '__main__':
    #dst = CityscapesDataSet("/home/wty/AllDataSet/CityScapes",
    #                        '/home/wty/AllDataSet/CityScapes/cityscapes_train_list.txt', scale=False)
    dst = CityscapesValDataSet("/home/wty/AllDataSet/CityScapes",
                            '/home/wty/AllDataSet/CityScapes/cityscapes_val_list.txt')
    trainloader = data.DataLoader(dst, batch_size=3)
    for i, data in enumerate(trainloader):
        imgs, labels, size, name = data
        if i == 0:
            print(name)
            print(size)
