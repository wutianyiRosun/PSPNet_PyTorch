import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from psp.model import PSPNet
from psp.loss import CrossEntropy2d
from psp.voc12_datasets import VOCDataSet
from psp.cityscapes_datasets import CityscapesDataSet
import matplotlib.pyplot as plt
import random
import timeit

start = timeit.default_timer()

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSPnet Network")

    # optimatization configuration
    parser.add_argument("--is-training", action="store_true", 
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default= 0.001, 
                        help="Base learning rate for training with polynomial decay.") #0.001
    parser.add_argument("--weight-decay", type=float, default= 0.0001, 
                        help="Regularisation parameter for L2-loss.")  # 0.0005
    parser.add_argument("--momentum", type=float, default= 0.9, 
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default= 0.9, 
                        help="Decay parameter to compute the learning rate.")

    # dataset information
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        help="voc12, cityscapes, or pascal-context.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")

    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--random-seed", type=int, default= 1234,
                        help="Random seed to have reproducible results.")
    parser.add_argument('--logFile', default='log.txt', 
                        help='File that stores the training and validation logs')
    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="choose gpu device.")


    return parser.parse_args()

args = get_arguments()

def configure_dataset_init_model(args):
    if args.dataset == 'voc12':

        args.batch_size = 10# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
        args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
        args.data_dir = '/home/wty/AllDataSet/VOC2012'   # Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/VOC2012/train_aug.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '473,473' #Comma-separated string with height and width of images
        args.num_classes = 21      #Number of classes to predict (including background)

        args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
        # saving model file and log record during the process of training

        #Where restore model pretrained on other dataset, such as COCO.")
        args.restore_from = './pretrained/MS_DeepLab_resnet_pretrained_COCO_init.pth'
        args.snapshot_dir = './snapshots/voc12/'          #Where to save snapshots of the model
        args.resume = './snapshots/voc12/psp_voc12_3.pth' #checkpoint log file, helping recovering training
        

    elif args.dataset == 'cityscapes':
        args.batch_size = 8   #Number of images sent to the network in one step, batch_size/num_GPU=2
        args.maxEpoches = 60 #epoch nums, 60 epoches is equal to 90k iterations, max iterations= maxEpoches*len(train)/batch_size')
        # 60x2975/2=89250 ~= 90k, single_GPU_batch_size=2
        args.data_dir = '/home/wty/AllDataSet/CityScapes'   # Path to the directory containing the PASCAL VOC dataset
        args.data_list = './dataset/list/Cityscapes/cityscapes_train_list.txt'  # Path to the file listing the images in the dataset
        args.ignore_label = 255     #The index of the label to ignore during the training
        args.input_size = '720,720' #Comma-separated string with height and width of images
        args.num_classes = 19      #Number of classes to predict (including background)

        args.img_mean = np.array((73.15835921, 82.90891754, 72.39239876), dtype=np.float32)
        # saving model file and log record during the process of training

        #Where restore model pretrained on other dataset, such as coarse cityscapes
        args.restore_from = './pretrained/resnet101_pretrained_for_cityscapes.pth'
        args.snapshot_dir = './snapshots/cityscapes/'          #Where to save snapshots of the model
        args.resume = './snapshots/cityscapes/psp_cityscapes_12_3.pth' #checkpoint log file, helping recovering training
       
    else:
        print("dataset error")



def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
    """
    This generator returns all the parameters of the net except for 
    the last classification layer. Note that for each batchnorm layer, 
    requires_grad is set to False in deeplab_resnet.py, therefore this function does not return 
    any batchnorm parameter
    """
    b = []
    if torch.cuda.device_count() == 1:
        b.append(model.conv1)
        b.append(model.bn1)
        b.append(model.layer1)
        b.append(model.layer2)
        b.append(model.layer3)
        b.append(model.layer4)
    else:
        b.append(model.module.conv1)
        b.append(model.module.bn1)
        b.append(model.module.layer1)
        b.append(model.module.layer2)
        b.append(model.module.layer3)
        b.append(model.module.layer4)
    
    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = []
    if torch.cuda.device_count() == 1:
        b.append(model.pspmodule.parameters())
        b.append(model.main_classifier.parameters())
    else:
        b.append(model.module.pspmodule.parameters())
        b.append(model.module.main_classifier.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i
            
            
def adjust_learning_rate(optimizer, i_iter, max_iter):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, i_iter, max_iter, args.power)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def netParams(model):
    '''
    Computing total network parameters
    Args:
       model: model
    return: total network parameters
    '''
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        #print(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters

def main():
    
    
    print("=====> Configure dataset and pretrained model")
    configure_dataset_init_model(args)
    print(args)

    print("    current dataset:  ", args.dataset)
    print("    init model: ", args.restore_from)
    print("=====> Set GPU for training")
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    print("=====> Random Seed: ", args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed) 

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True

    print("=====> Building network")
    model = PSPNet(num_classes=args.num_classes)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

   
    print("=====> Loading init weights,  pretrained COCO for VOC2012, and pretrained Coarse cityscapes for cityscapes")
    saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        #Scale.layer5.conv2d_list.3.weight
        #print(saved_state_dict[i].size())
        i_parts = i.split('.')
        #print('i_parts:  ', i_parts)
        if  not i_parts[1]=='layer5':  #init model pretrained on COCO, class name=21, layer5 is ASPP
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            #print('copy {}'.format('.'.join(i_parts[1:])))
    model.load_state_dict(new_params)

    
    if args.cuda:
        if torch.cuda.device_count()>1:
            print("torch.cuda.device_count()=",torch.cuda.device_count())
            model = torch.nn.DataParallel(model).cuda()  #multi-card data parallel
        else:
            print("single GPU for training")
            model = model.cuda()  #1-card data parallel
    start_epoch=0
    
    print("=====> Whether resuming from a checkpoint, for continuing training")
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"] 
            model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    model.train()
    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    print('=====> Computing network parameters')
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
 
    print("=====> Preparing training data")
    if args.dataset == 'voc12':
        trainloader = data.DataLoader(VOCDataSet(args.data_dir, args.data_list, max_iters=None, crop_size=input_size, 
                                                 scale=args.random_scale, mirror=args.random_mirror, mean=args.img_mean), 
                                      batch_size= args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    elif args.dataset == 'cityscapes':
        trainloader = data.DataLoader(CityscapesDataSet(args.data_dir, args.data_list, max_iters=None, crop_size=input_size, 
                                                 scale=args.random_scale, mirror=args.random_mirror, mean=args.img_mean), 
                                      batch_size = args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    else:
        print("dataset error")

    optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': args.learning_rate }, 
                {'params': get_10x_lr_params(model), 'lr': 10*args.learning_rate}], 
                lr=args.learning_rate, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer.zero_grad()


    
    logFileLoc = args.snapshot_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t\t%s" % ('iter', 'Loss(train)\n'))
    logger.flush()

    print("=====> Begin to train")
    train_len=len(trainloader)
    print("  iteration numbers  of per epoch: ", train_len)
    print("  epoch num: ", args.maxEpoches)
    print("  max iteration: ", args.maxEpoches*train_len)
    for epoch in range(start_epoch, int(args.maxEpoches)):
        
        for i_iter, batch in enumerate(trainloader,0): #i_iter from 0 to len-1
            #print("i_iter=", i_iter, "epoch=", epoch)
            images, labels, _, _ = batch
            images = Variable(images).cuda()

            optimizer.zero_grad()
            lr = adjust_learning_rate(optimizer, i_iter+epoch*train_len, 
                    max_iter = args.maxEpoches * train_len)
            pred = model(images)
            print("model output size: ", pred.size())
            loss = loss_calc(pred, labels)
            loss.backward()
            optimizer.step()
            
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}  lr: {:.5f}".format(epoch, i_iter, train_len, loss.data[0], lr))
            logger.write("Epoch[{}]({}/{}):     Loss: {:.10f}      lr: {:.5f}\n".format(epoch, i_iter, train_len, loss.data[0], lr))
            logger.flush()
                
        print("=====> saving model")
        state={"epoch": epoch+1, "model": model.state_dict()}
        torch.save(state, osp.join(args.snapshot_dir, 'psp_'+str(args.dataset)+"_"+str(epoch)+'.pth'))


    end = timeit.default_timer()
    print( float(end-start)/3600, 'h')
    logger.write("total training time: {:.2f} h\n".format(float(end-start)/3600))
    logger.close()


if __name__ == '__main__':
    main()
