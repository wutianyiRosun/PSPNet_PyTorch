import torch
import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import cv2
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data
from psp.model import PSPNet
from psp.voc12_datasets import VOCDataTestSet
from psp.cityscapes_datasets import CityscapesTestDataSet
from collections import OrderedDict
from utils.colorize_mask import cityscapes_colorize_mask, VOCColorize
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSPnet")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        help="voc12, cityscapes, or pascal-context")

    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="3",
                        help="choose gpu device.")
    return parser.parse_args()

def configure_dataset_model(args):
    if args.dataset == 'voc12':
        args.data_dir ='/home/wty/AllDataSet/VOC2012'  #Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/VOC2012/test.txt'  #Path to the file listing the images in the dataset
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) 
        #RBG mean, first subtract mean and then change to BGR
        args.ignore_label = 255   #The index of the label to ignore during the training
        args.num_classes = 21  #Number of classes to predict (including background)
        args.restore_from = './snapshots/voc12/psp_voc12_14.pth'  #Where restore model parameters from
        args.save_segimage = True
        args.seg_save_dir = "./result/test/VOC2012"
        args.corp_size =(505, 505)
    
    elif args.dataset == 'cityscapes':
        args.data_dir ='/home/wty/AllDataSet/CityScapes'  #Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/Cityscapes/cityscapes_test_list.txt'  #Path to the file listing the images in the dataset
        args.img_mean = np.array((73.15835921, 82.90891754, 72.39239876), dtype=np.float32)
        #RBG mean, first subtract mean and then change to BGR
        args.ignore_label = 255   #The index of the label to ignore during the training
        args.f_scale = 1  #resize image, and Unsample model output to original image size, label keeps
        args.num_classes = 19  #Number of classes to predict (including background)
        args.restore_from = './snapshots/cityscapes/psp_cityscapes_59.pth'  #Where restore model parameters from
        args.save_segimage = True
        args.seg_save_dir = "./result/test/Cityscapes"
    else:
        print("dataset error")

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
       You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it 
       without DataParallel. You can either add a nn.DataParallel temporarily in your network for loading purposes, or you can 
       load the weights file, create a new ordered dict without the module prefix, and load it back 
    """
    state_dict_new = OrderedDict()
    #print(type(state_dict))
    for k, v in state_dict.items():
        #print(k)
        name = k[7:] # remove the prefix module.
        # My heart is borken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
    return state_dict_new


def main():
    args = get_arguments()
    print("=====> Configure dataset and model")
    configure_dataset_model(args)
    print(args)

    print("=====> Set GPU for training")
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    model = PSPNet(num_classes=args.num_classes)
    
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict( convert_state_dict(saved_state_dict["model"]) )

    model.eval()
    model.cuda()
    if args.dataset == 'voc12':
        testloader = data.DataLoader(VOCDataTestSet(args.data_dir, args.data_list, crop_size=(505, 505),mean= args.img_mean), 
                                    batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=(505, 505), mode='bilinear')
        voc_colorize = VOCColorize()
    elif args.dataset == 'cityscapes':
        testloader = data.DataLoader(CityscapesTestDataSet(args.data_dir, args.data_list, f_scale= args.f_scale, mean= args.img_mean), 
                                    batch_size=1, shuffle=False, pin_memory=True) # f_sale, meaning resize image at f_scale as input
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')  #size = (h,w)
    else:
        print("dataset error")

    data_list = []

    if args.save_segimage:
        if not os.path.exists(args.seg_save_dir):
            os.makedirs(args.seg_save_dir)
    print("======> test set size:", len(testloader))
    for index, batch in enumerate(testloader):
        print('%d processd'%(index))
        image, size, name = batch
        size = size[0].numpy()
        output = model(Variable(image, volatile=True).cuda())

        output = interp(output).cpu().data[0].numpy()

        if args.dataset == 'voc12':
            print(output.shape)
            print(size)
            output = output[:,:size[0],:size[1]]
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            if args.save_segimage:
                seg_filename = os.path.join(args.seg_save_dir, '{}.png'.format(name[0]))
                color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                color_file.save(seg_filename)

        elif args.dataset == 'cityscapes':
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            if args.save_segimage:
                output_color = cityscapes_colorize_mask(output)
                output = Image.fromarray(output)
                output.save('%s/%s.png'% (args.seg_save_dir, name[0]))
                output_color.save('%s/%s_color.png'%(args.seg_save_dir, name[0]))
        else:
            print("dataset error")


if __name__ == '__main__':
    main()
