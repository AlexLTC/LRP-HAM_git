# --------------------------------------------------------
# Tensorflow LRP-HAI
# Licensed under The MIT License [see LICENSE for details]
# Written by Chang Hsiao-Chien
# Written by Aleksis Pirinen
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths_drl
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import time, os, sys
import numpy as np
import sys
from time import sleep
from utils.logger import setup_logger

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.P4 import P4

import datetime

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
# session = tf.Session(config=config)
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

# 想改res101變p4, 要改config.py

def parse_args():
    """
  Parse input arguments
  """

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Train a LRP-HAI network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='./experiments/cfgs/LRP-HAI-P4.yml', type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model weights',
                        default='./fr-rcnn-weights/P4/res101/voc_2007_trainval/default_new_throat/res101_faster_rcnn_iter_70000.ckpt',
                        type=str)
    # default = '/media/data/LRP-HAI/fr-rcnn-voc2007-2012-trainval/vgg16_faster_rcnn_iter_180000.ckpt'

    parser.add_argument('--save', dest='save_path',
                        help='path for saving model weights',
                        default='./TEST/L-RPN-HAM/pascal_voc/P4/res101/L-RPN-HAM-test/',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        #default='cell_train', type=str)
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--imdbval', dest='imdbval_name',
                        help='dataset to validate on',
                        #default='cell_val', type=str)
                        default='voc_2007_test', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=110000, type=int)#110000
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res101', type=str)
    parser.add_argument('--det_start', dest='det_start',
                        help='-1: dont train detector; >=0: train detector onwards',
                        default=40000, type=int)
    parser.add_argument("--alpha", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Activate alpha mode.")
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def combined_roidb(imdb_names):
    """
  Combine multiple roidbs
  """

    def get_roidb(imdb_name):# 處理名子 imdb image database
        imdb = get_imdb(imdb_name)  # imdb_name = 'voc_2007_trainval'
        # 由给出的以上参数确定的数据集的路径为self._data_path =$CODE_DIR / faster - rcnn / data / VOCdevkit2007 / VOC2007
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)  # train.proposal_method == 'gt'
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)#"Returns a roidb (Region of Interest database) for use in training."
        # roidb 每張照片的詳細數據

        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]  # training set
    # split 用+分隔字句 ,將imdb_name的資料集依序輸入到roidbs中
    # s=voc_2007_trainval,imdb_names=voc_2007_trainval
    roidb = roidbs[0]
    # {'boxes': array([[ 58,  32, 164, 151],
    # [137, 145, 184, 223],
    # [ 81, 103,  87, 115],
    # [107,  88, 131, 145]], dtype=uint16), 'gt_classes': array([1, 2, 3, 3], dtype=int32),
    # 'gt_overlaps': <4x4 sparse matrix of type '<class 'numpy.float32'>'
	# with 4 stored elements in Compressed Sparse Row format>,
    # 'flipped': False, 'seg_areas': array([12840.,  3792.,    91.,  1450.],
    # dtype=float32), 'image': '/home/dennischang/LRP-HAI/data/VOCdevkit2007/VOC2007/JPEGImages/(9).jpg',
    # 'width': 224, 'height': 224, 'max_classes': array([1, 2, 3, 3]), 'max_overlaps': array([1., 1., 1., 1.], dtype=float32)}

    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)  # 加入第r個資料在後
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb
    # roidb 每張照片的詳細數據 gt大小 照片大小等


if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)# LRP-HAI-P4.yml
    if args.set_cfgs is not None:# None
        cfg_from_list(args.set_cfgs)

    np.random.seed(cfg.RNG_SEED)# 3 ,在np.random.seed後的是某一組的隨機值,再次在叫3還是會輸出一樣的值


    # train set
    # imdb, roidb = combined_roidb(args.imdb_name)是数据准备的核心部分。
    # imdb {'__background__': 0, 'epiglottis': 1, 'arytenoidcartilages': 2, 'vocalfolds': 3}
    # roidb则包含了训练网络所需要的所有信息
    imdb, roidb = combined_roidb(args.imdb_name)# voc_2007_trainval

    # Set class names in config file based on IMDB
    class_names = imdb.classes  # array([1, 2])
    # class_names <class 'tuple'>: ('__background__', 'epiglottis', 'arytenoidcartilages', 'vocalfolds')
    cfg_from_list(['CLASS_NAMES', [class_names]])# 此處是將數值輸入

    if args.alpha:  # true
        cfg_from_list(['LRP_HAI.ALPHA', True])

    # Update config to match start of training detector
    cfg_from_list(['LRP_HAI_TRAIN.DET_START', args.det_start])  # 40000

    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag, args.save_path)
    # args.save_path LRP - HAI / TEST
    # '/home/dennischang/LRP-HAI/TEST/L-RPN-HAM/pascal_voc_喉鏡0916/P4/res101/L-RPN-HAM-1/output/res101_LRP_HAI/voc_2007_trainval/'

    # 將資料寫在log_train.txt上
    logger = setup_logger("LRP-HAI", save_dir=args.save_path, filename="log_train.txt")
    logger.info('Called with args:')
    logger.info(args)
    logger.info('Using attention alpha:')
    logger.info(cfg.LRP_HAI.ALPHA)
    logger.info('Using config:\n{}'.format(pprint.pformat(cfg)))
    logger.info('{:d} roidb entries'.format(len(roidb)))
    logger.info('Output will be saved to `{:s}`'.format(output_dir))
    
    # make tensorboard output directory
    tb_dir = get_output_tb_dir(imdb, "ARM")
    tb_dir = os.path.join(tb_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger.info('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED  # true
    cfg.TRAIN.USE_FLIPPED = False
    _, valroidb = combined_roidb(args.imdbval_name)  # voc_2007_test
    logger.info('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip

    if cfg.P4:
        # load network
        if args.net == 'res50':
            net = P4(num_layers=50)
        elif args.net == 'res101':
            net = P4(num_layers=101)
        else:
            raise NotImplementedError
    else:
        # load network
        if args.net == 'vgg16':
            net = vgg16()
        elif args.net == 'res50':
            net = resnetv1(num_layers=50)
        elif args.net == 'res101':
            net = resnetv1(num_layers=101)
        else:
            raise NotImplementedError

    # 實際開始訓練
    train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,pretrained_model=args.weight,
              max_iters=args.max_iters)


    # net res101 p4
    # imdb <class 'dict'>: {'__background__': 0, 'epiglottis': 1, 'arytenoidcartilages': 2, 'vocalfolds': 3}
    # roidb 訓練集資料
    # valroidb 測試集資料
    # output_dir 儲存訓練權重的地方
    # pretrained_model : faster rcnn 預訓練權重
    #
