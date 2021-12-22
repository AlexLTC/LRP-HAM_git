# --------------------------------------------------------
# Tensorflow LRP-HAI
# Licensed under The MIT License [see LICENSE for details]
# Partially written by Chang Hsiao-Chien
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# 調控glimpse部份的輸出，由/LRP-HAI/lib_drl/model/factor.py/save_visualization處調整
#
# search "set video path" to find realtime visualization code
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths_drl
from model.realtime_factory import run_LRP_HAI, print_timings, get_image_blob
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import numpy as np
from time import sleep
from utils.logger import setup_logger
from utils.timer import Timer
import cv2
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.P4 import P4
import pyrealsense2
from realsense_depth import *


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

    parser = argparse.ArgumentParser(description='Test a LRP-HAI network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', 
                        #default='experiments/cfgs/LRP-HAI-P4.yml', type=str)
                        default='experiments/cfgs/LRP-HAI-P4.yml', type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        #default='/media/data/LRP-HAI/experiments/drl-model-2/P4/res101/drl-model-2-1/output/res101_LRP_HAI/cell_train/res101_LRP_HAI_iter_110000.ckpt',
                        #default='/home/dennischang/LRP-HAI/TEST/L-RPN-HAM/pascal_voc_喉鏡（副本）/P4/res101/L-RPN-HAM-1/output/res101_LRP_HAI/voc_2007_trainval/res101_LRP_HAI_iter_110000.ckpt',
                        #default='/home/user/xuus_blue/LRP-HAM/HAM-weights_Main0/polyp/P4/res101/output/P4/res101/polyp_2007_trainval/res101_LRP_HAI_iter_110000.ckpt',
                        #default='/home/user/xuus_blue/LRP-HAM/HAM-weights_Main0/pascal_voc/P4/res101/output/P4/res101/voc_2007_trainval/res101_LRP_HAI_iter_110000.ckpt',
                        #default='/home/user/xuus_blue/LRP-HAM/HAM-weights_Main0/throat/P4/res101/output/P4/res101/throat_2007_trainval/res101_LRP_HAI_iter_110000.ckpt',
			default='/media/xuus/A45ED35B5ED324B8/LRP-HAM/HAM-weights/throat_uvula/P4/res101/output/P4/res101/throat_uvula_2007_trainval/res101_LRP_HAI_iter_110000.ckpt',
                        type=str)# 改這裡
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        # default='voc_2007_test', type=str)
                        # default='cell_val', type=str)
			# default='throat_2007_test', type=str)
			# default='polyp_2007_test', type=str)
                        default='throat_uvula_2007_test', type=str)# 改這裡
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res101', type=str)
                        #default='res101', type=str)
    parser.add_argument('--nbr_fix', dest='nbr_fix',
                        help='0: auto-stop, > 0 run LRP-HAI exactly nbr_fix steps',
                        default=0, type=int)
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


def im_detect(sess, net, im, timers, im_idx=None):
    # Setup image blob
    blobs = {}
    blobs['data'], im_scales, blobs['im_shape_orig'] = get_image_blob(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]])

    # Run LRP-HAI
    # new add cls_det for showing bouding box on frame
    cls_det, scores, pred_bboxes, timers, _\
        = run_LRP_HAI(sess, net, blobs, timers, 'test', cfg.LRP_HAI_TEST.BETA, im_idx, alpha=cfg.LRP_HAI.ALPHA)

    return cls_det, scores, pred_bboxes, timers, _


if __name__ == '__main__':
    args = parse_args()

    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)

    # if has model, get the name from it
    # if does not, then just use the initialization weights
    if args.model:
        filename = os.path.splitext(os.path.basename(args.model))[0]
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]

    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    # This extra_string used by me (Aleksis) when running code on two
    # different machines, for convenience
    extra_string = ''
    # print(args.imdb_name)
    if args.imdb_name == 'voc_2012_test':
        extra_string += '_test'
    # print(args.imdb_name + extra_string)
    # sleep(100)
    imdb = get_imdb(args.imdb_name + extra_string)
    imdb.competition_mode(args.comp_mode)

    # new add
    output_dir = get_output_dir(imdb, filename)
    logger = setup_logger("LRP-HAI", save_dir=output_dir, filename="log_test.txt")
    logger.info('Called with args:')
    logger.info(args)

    # Set class names in config file based on IMDB
    class_names = imdb.classes
    cfg_from_list(['CLASS_NAMES', [class_names]])
    if args.alpha:
        cfg_from_list(['LRP_HAI.ALPHA', True])
    else:
        cfg_from_list(['LRP_HAI.ALPHA', False])

    # VISUALIZE IMAGE
    cfg_from_list(['LRP_HAI_TEST.DO_VISUALIZE', True])

    # Specify if run LRP-HAI in auto mode or a fix number of iterations
    cfg_from_list(['LRP_HAI_TEST.NBR_FIX', args.nbr_fix])
    logger.info('Using config:\n{}' .format(pprint.pformat(cfg)))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # tfconfig.gpu_options.visible_device_list = '0'
    # 最多占gpu资源的70%
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.7

    # Set the random seed for tensorflow
    tf.set_random_seed(cfg.RNG_SEED)

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
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

    # load model
    net.create_architecture("TEST", imdb.num_classes, tag='default',
                            anchor_sizes=cfg.ANCHOR_SIZES,
                            anchor_strides=cfg.ANCHOR_STRIDES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

    net.build_LRP_HAI_network(False)

    if args.model:
        logger.info('Loading model check point from {:s}'.format(args.model))
        tf.train.Saver().restore(sess, args.model)
        logger.info('Loaded.')
    else:
        logger.info('Loading initial weights from {:s}'.format(args.weight))
        sess.run(tf.global_variables_initializer())
        logger.info('Loaded.')

    # demo
    # Visualize search trajectories?
    do_visualize = cfg.LRP_HAI_TEST.DO_VISUALIZE

    """
    # 改這裡
    load_dir = "demo_image/throat_uvula_from_net"
    imgs = os.listdir(load_dir)
    _t = {'im_detect': Timer(), 'misc': Timer(), 'total_time': Timer()}
    _t_LRP_HAI = {'init': Timer(), 'fulltraj': Timer(),
                  'upd-obs-vol': Timer(), 'upd-seq': Timer(), 'upd-rl': Timer(),
                  'action-rl': Timer(), 'coll-traj': Timer()}
    imgs.sort(key= lambda x:int(x[:-4]))
    for i in range(len(imgs)):

        # Need to know image index if performing visualizations
        if do_visualize:
            im_idx = i#
        else:
            im_idx = None

        # Detect!
        im = cv2.imread(os.path.join(load_dir, imgs[i]))
        _t['im_detect'].tic()
        scores, boxes, _t_LRP_HAI, _ = im_detect(sess, net, im, _t_LRP_HAI,
                                                     im_idx)
        _t['im_detect'].toc()

        logger.info('\nim_detect: {:d}/{:d} {:.3f}s'.format(i + 1, len(imgs), _t['im_detect'].average_time))

    """


    # set video path
    # for realsense realtime visualization
    dc = DepthCamera()
    
    _t = {'im_detect': Timer(), 'misc': Timer(), 'total_time': Timer()}
    _t_LRP_HAI = {'init': Timer(), 'fulltraj': Timer(),
                  'upd-obs-vol': Timer(), 'upd-seq': Timer(), 'upd-rl': Timer(),
                  'action-rl': Timer(), 'coll-traj': Timer()}
    
    while True:
        ret, depth_frame, color_frame = dc.get_frame()
        if ret :
            _t['im_detect'].tic()
    
            # need im_idx to run do_visualization
            cls_det, scores, boxes, _t_LRP_HAI, _ = im_detect(sess, net, color_frame, _t_LRP_HAI, im_idx=1)
            if len(cls_det) != 0:
                print("x1:{}, y1:{}, x2:{}, y2:{}".format(cls_det[0], cls_det[1], cls_det[2], cls_det[3]))
                cv2.rectangle(color_frame, (cls_det[0], cls_det[1]), (cls_det[2], cls_det[3]), (255, 0, 0), 2)
    
                # show box center point depth
                pointX = int(np.round((cls_det[2] + cls_det[0])/2))
                pointY = int(np.round((cls_det[3] + cls_det[1])/2))
                point = (pointX, pointY)
                cv2.circle(color_frame, point, 4, (0, 0, 255))
                print('type: ', type(point[1]))
                print('number', point[1])
                distance = depth_frame[point[1], point[0]]
                cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 0), 2)
    
    
            # # sharpen&resize image
            # sigma = 20 # sigma bigger than 20 will severely slow down showing
            # blur_img = cv2.GaussianBlur(color_frame, (0,0), sigma)
            # color_frame = cv2.addWeighted(color_frame, 1.5, blur_img, -0.5, 0)
            # color_frame = cv2.resize(color_frame, (1280, 720))
    
    
            cv2.imshow('Color Frame', color_frame)
    
            key = cv2.waitKey(1)
            if key == 27:
                break
            _t['im_detect'].toc()
    
            # logger.info('\nim_detect: {:d}/{:d} {:.3f}s'.format(i + 1, len(imgs), _t['im_detect'].average_time))
        else :
            print('ret is false')
            break
    
    cv2.destroyAllWindows()
    
    '''
    # for video use
    cap = cv2.VideoCapture('/media/xuus/A45ED35B5ED324B8/alex/rt_test_mp4/Produce22.mp4')
    if not cap.isOpened:
        print('failed open video')

    _t = {'im_detect': Timer(), 'misc': Timer(), 'total_time': Timer()}
    _t_LRP_HAI = {'init': Timer(), 'fulltraj': Timer(),
                  'upd-obs-vol': Timer(), 'upd-seq': Timer(), 'upd-rl': Timer(),
                  'action-rl': Timer(), 'coll-traj': Timer()}

    while(cap.isOpened):

        # Detect!
        ret, frame = cap.read()
        # flip the original picture to fit the network
        frame = cv2.flip(frame, 0)

        if ret:
            _t['im_detect'].tic()

            # need im_idx to run do_visualization
            cls_det, scores, boxes, _t_LRP_HAI, _ = im_detect(sess, net, frame, _t_LRP_HAI, im_idx=1)
            if len(cls_det) != 0:
                cv2.rectangle(frame, (cls_det[0], cls_det[1]), (cls_det[2], cls_det[3]), (255, 0, 0), 2)
                cv2.putText(frame, 'uvula', (int(cls_det[2]), int(cls_det[3]) + 10) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # # sharpen&resize image
            # sigma = 20 # sigma bigger than 20 will severely slow down showing
            # blur_img = cv2.GaussianBlur(frame, (0,0), sigma)
            # frame = cv2.addWeighted(frame, 1.5, blur_img, -0.5, 0)
            frame = cv2.resize(frame, (1200, 900))

            cv2.imshow('Frame', frame)

            if cv2.waitKey(1) == ord('q'):
                break
            _t['im_detect'].toc()

            # logger.info('\nim_detect: {:d}/{:d} {:.3f}s'.format(i + 1, len(imgs), _t['im_detect'].average_time))
        else :
            print('ret is false')
            break

    cap.release()

    cv2.destroyAllWindows()
    '''

	# alex add for output img's (x,y,w,h)
        #with open('/home/user/Desktop/alex/real_demo_test', 'a') as f :
        #    f.write('img num: ' + str(i + 1) + '\n')
	#    f.write('\n') 


        # print_timings(_t_LRP_HAI) # uncomment for some timing details!

    sess.close()
