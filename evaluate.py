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
from psp.voc12_datasets import VOCDataSet
from psp.cityscapes_datasets import CityscapesValDataSet
from collections import OrderedDict
from utils.colorize_mask import cityscapes_colorize_mask, VOCColorize
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
import time


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
        args.data_list = './dataset/list/VOC2012/val.txt'  #Path to the file listing the images in the dataset
        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32) 
        #RBG mean, first subtract mean and then change to BGR
        args.ignore_label = 255   #The index of the label to ignore during the training
        args.num_classes = 21  #Number of classes to predict (including background)
        args.restore_from = './snapshots/voc12/psp_voc12_10.pth'  #Where restore model parameters from
        args.save_segimage = True
        args.seg_save_dir = "./result/val/VOC2012"
        args.corp_size =(505, 505)
    
    elif args.dataset == 'cityscapes':
        args.data_dir ='/home/wty/AllDataSet/CityScapes'  #Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/Cityscapes/cityscapes_val_list.txt'  #Path to the file listing the images in the dataset
        args.img_mean = np.array((73.15835921, 82.90891754, 72.39239876), dtype=np.float32)
        #RBG mean, first subtract mean and then change to BGR
        args.ignore_label = 255   #The index of the label to ignore during the training
        args.num_classes = 19  #Number of classes to predict (including background)
        args.f_scale = 1  #resize image, and Unsample model output to original image size
        args.restore_from = './snapshots/cityscapes/psp_cityscapes_20.pth'  #Where restore model parameters from
        #args.restore_from = './pretrained/resnet101_pretrained_for_cityscapes.pth'  #Where restore model parameters from
        args.save_segimage = True
        args.seg_save_dir = "./result/val/Cityscapes"
    else:
        print("dataset error when configuring dataset and model")

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

def get_iou(data_list, class_num, save_path=None):
    from multiprocessing import Pool 
    from psp.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('meanIOU: ' + str(aveJ) + '\n')
            f.write(str(j_list)+'\n')
            f.write(str(M)+'\n')

def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
                    (0.5,0.75,0),(0,0.25,0.5)]
    cmap = colors.ListedColormap(colormap)
    bounds=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()

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
    if args.dataset == 'voc12':
        model.load_state_dict( convert_state_dict(saved_state_dict["model"]) )
    elif args.dataset == 'cityscapes':
        #model.load_state_dict(saved_state_dict["model"])
        model.load_state_dict( convert_state_dict(saved_state_dict["model"]) )
    else:
        print("dataset error when loading model file")

    model.eval()
    model.cuda()
    if args.dataset == 'voc12':
        testloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, crop_size=(505, 505), 
                                                mean= args.img_mean, scale=False, mirror=False), 
                                    batch_size=1, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=(505, 505), mode='bilinear')
        voc_colorize = VOCColorize()
    elif args.dataset == 'cityscapes':
        testloader = data.DataLoader(CityscapesValDataSet(args.data_dir, args.data_list, f_scale=args.f_scale, mean= args.img_mean), 
                                    batch_size=1, shuffle=False, pin_memory=True) # f_sale, meaning resize image at f_scale as input
        interp = nn.Upsample(size=(1024, 2048), mode='bilinear')  #size = (h,w)
    else:
        print("dataset error when configure DataLoader")

    data_list = []

    if args.save_segimage:
        if not os.path.exists(args.seg_save_dir):
            os.makedirs(args.seg_save_dir)

    for index, batch in enumerate(testloader):
        image, label, size, name = batch
        #print("label.size:", label.size())
        #print("model input image size:", image.size())
        size = size[0].numpy()
        start_time = time.time()
        output = model(Variable(image, volatile=True).cuda())

        output = interp(output).cpu().data[0].numpy()

        time_taken = time.time() - start_time;
        print('%d processd,  time: %.3f'%(index, time_taken))

        if args.dataset == 'voc12':
            output = output[:,:size[0],:size[1]]
            gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            data_list.append([gt.flatten(), output.flatten()])
            if args.save_segimage:
                seg_filename = os.path.join(args.seg_save_dir, '{}.png'.format(name[0]))
                color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
                color_file.save(seg_filename)

        elif args.dataset == 'cityscapes':
            gt = np.asarray(label[0].numpy(), dtype=np.int)
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            data_list.append([gt.flatten(), output.flatten()])
            if args.save_segimage:
                output_color = cityscapes_colorize_mask(output)
                output = Image.fromarray(output)
                output.save('%s/%s.jpg'% (args.seg_save_dir, name[0]))
                output_color.save('%s/%s_color.png'%(args.seg_save_dir, name[0]))
        else:
            print("dataset error")

    get_iou(data_list, args.num_classes)


if __name__ == '__main__':
    main()
