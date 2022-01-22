# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np

from layer_utils.roi_align import roi_align
from layer_utils.snippets import generate_anchors_pre, generate_anchors_pre_tf
from layer_utils.proposal_layer import proposal_layer, proposal_layer_tf
from layer_utils.proposal_top_layer import proposal_top_layer, proposal_top_layer_tf
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from utils.visualization import draw_bounding_boxes

from model.config import cfg

import cv2


class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._event_summaries = {}
        self._variables_to_fix = {}

    def _add_gt_image(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] / self._im_info[2]))
        self._gt_image = tf.reverse(resized, axis=[-1])

    def _add_gt_image_summary(self):
        # use a customized visualization function to visualize the boxes
        if self._gt_image is None:
            self._add_gt_image()
        image = tf.py_func(draw_bounding_boxes,
                           [self._gt_image, self._gt_boxes, self._im_info],
                           tf.float32, name="gt_boxes")

        return tf.summary.image('GROUND_TRUTH', image)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)  # [batch, channel, row, column]
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])  # [1, row*col/2, channel, 2]
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_top_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_top_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._anchors],
                                              [tf.float32, tf.float32], name="proposal_top")

            rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
            rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

        return rois, rpn_scores

    def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
        with tf.variable_scope(name) as scope:
            if cfg.USE_E2E_TF:
                rois, rpn_scores = proposal_layer_tf(
                    rpn_cls_prob,
                    rpn_bbox_pred,
                    self._im_info,
                    self._mode,
                    self._anchors,
                    self._num_anchors
                )
            else:
                rois, rpn_scores = tf.py_func(proposal_layer,
                                              [rpn_cls_prob, rpn_bbox_pred, self._im_info, self._mode,
                                               self._anchors],
                                              [tf.float32, tf.float32], name="proposal")

            rois.set_shape([None, 5])
            rpn_scores.set_shape([None, 1])

        return rois, rpn_scores

    def _crop_pool_layer(self, bottom, rois, _im_info, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            img_h, img_w = tf.cast(_im_info[0], tf.float32), tf.cast(_im_info[1], tf.float32)
            # N = tf.shape(rois)[0]
            _, x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)
            pre_pool_size = cfg.POOLING_SIZE * 2
            cropped_roi_features = tf.image.crop_and_resize(bottom, normalized_rois,
                                                            box_ind=tf.to_int32(batch_ids),
                                                            crop_size=[pre_pool_size, pre_pool_size],
                                                            name='CROP_AND_RESIZE'
                                                            )
            roi_features = slim.max_pool2d(cropped_roi_features, [2, 2], stride=2, padding='SAME')
        return roi_features

    def _dropout_layer(self, bottom, name, ratio=0.5):
        return tf.nn.dropout(bottom, ratio, name=name)

    def _anchor_target_layer(self, rpn_cls_score, name):
        all_anchors = self._anchors
        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
                anchor_target_layer,
                [rpn_cls_score, self._gt_boxes, self._im_info, all_anchors, self._num_anchors],
                [tf.float32, tf.float32, tf.float32, tf.float32],
                name="anchor_target")

            rpn_labels.set_shape([1, 1, None, None])
            rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
            rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

            rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

        return rpn_labels

    def _proposal_target_layer(self, rois, roi_scores, name):
        with tf.variable_scope(name) as scope:
            rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer,
                [rois, roi_scores, self._gt_boxes, self._num_classes],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
                name="proposal_target")

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            labels = tf.to_int32(labels, name="to_int32")
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = labels
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)
            return rois, roi_scores

    def _anchor_component(self, net_conv):
        """
            all_anchors: (-1, 4)
        """
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            print('anchor_strides: ', self._anchor_strides)
            height, width = tf.shape(net_conv)[1], tf.shape(net_conv)[2]
            if cfg.USE_E2E_TF:
                anchors, anchor_length = generate_anchors_pre_tf(
                    height,
                    width,
                    self._anchor_strides[0],
                    self._anchor_sizes,
                    self._anchor_ratios
                )
            else:
                anchors, anchor_length = tf.py_func(generate_anchors_pre,
                                                    [height, width,
                                                     self._anchor_strides[0],
                                                     self._anchor_sizes,
                                                     self._anchor_ratios],
                                                    [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        # net_conv output : Tensor("build_pyramid/fuse_P4/BiasAdd:0", shape=(1, ?, ?, 256), dtype=float32))"")
        net_conv = self._image_to_head(is_training)  # P4 information (from lib_fr/nets/P4 function _image_to_head)
        with tf.variable_scope(self._scope, self._scope):
            # generate_anchors
            assert len(cfg.ANCHOR_STRIDES) == 1, "Non-FPN should have a single ANCHOR_STRIDE"
            self._anchor_component(net_conv)

            # region proposal network
            rois = self._region_proposal(net_conv, is_training, initializer)
            # region of interest pooling
            if cfg.POOLING_MODE == 'roi_pooling':
                pool5 = self._crop_pool_layer(net_conv, rois, self._im_info, "pool5")
            elif cfg.POOLING_MODE == 'roi_align':
                pool5 = roi_align(net_conv, rois, self._anchor_strides[0], cfg.POOLING_SIZE)
            else:
                raise NotImplementedError

        # fc7 output: Tensor("build_fc_layers/fc7/Relu:0", shape=(256, 1024), dtype=float32))
        fc7 = self._head_to_tail(pool5, is_training)
        with tf.variable_scope(self._scope, self._scope):
            # region classification
            cls_prob, bbox_pred = self._region_classification(fc7, is_training,
                                                              initializer, initializer_bbox)

        #----------------DA & DA_Conv Start----------------------------
        # Get Caffe prototext from the link bellow
        # https://github.com/yuhuayc/da-faster-rcnn/blob/master/models/da_faster_rcnn/train.prototxt
        #
        # Get tf GRL from the link bellow
        # https://github.com/pumpikano/tf-dann/blob/master/MNIST-DANN.ipynb 
        #
        # dc_ip3 output: 
        # Tensor("instance-level_DA/dc_ip3/Relu:0", shape=(256, 1), dtype=float32)
        dc_ip3 = self._tail_to_da(fc7, is_training)
        # da_score_ss output: 
        # Tensor("image-level_DA/da_conv_ss/Relu:0", shape=(1, ?, ?, 2), dtype=float32)
        da_score_ss = self._head_to_daConv(net_conv, is_training)
        # print("dc_ip3 looks like:")
        # print(dc_ip3)
        # print("da_score_ss looks like:")
        # print(da_score_ss)
        dc_ip3, da_score_ss = self._add_da_components(dc_ip3, da_score_ss)

        #---------------DA & DA_Conv End-------------------------------
        
        self._score_summaries.update(self._predictions)

        return rois, cls_prob, bbox_pred

    def _add_da_components(self, dc_ip3, da_score_ss):
        # _prediction[] append da components
        self._predictions['dc_ip3'] = dc_ip3
        self._predictions['da_score_ss'] = da_score_ss

        return dc_ip3, da_score_ss

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = box_diff
        # in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        in_loss_box = bbox_inside_weights * in_loss_box
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(
            out_loss_box,
            axis=dim
        ))
        return loss_box

    def _add_losses(self, sigma_rpn=3.0):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            # RPN, class loss
            rpn_cls_score = self._predictions['rpn_cls_score_reshape']
            rpn_label = self._anchor_targets['rpn_labels']

            rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])
            rpn_label = tf.reshape(rpn_label, [-1])
            rpn_select = tf.where(tf.not_equal(rpn_label, -1))
            rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
            rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
            rpn_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

            # RPN, bbox loss
            rpn_bbox_pred = self._predictions['rpn_bbox_pred']
            rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
            rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
            rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
            rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                    rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

            # RCNN, class loss
            cls_score = self._predictions["cls_score"]
            label = tf.reshape(self._proposal_targets["labels"], [-1])
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)

            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            self._losses['rpn_cross_entropy'] = rpn_cross_entropy
            self._losses['rpn_loss_box'] = rpn_loss_box
            
            # --------------------DA Part Start-------------------------
            #
            # instance-level loss
            dc_ip3 = self._predictions['dc_ip3']
            dc_label_resize = self._label_resize_layer(dc_ip3)
            dc_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=dc_ip3, 
                                                            labels=dc_label_resize))
            # image-level loss
            da_score_ss = self._predictions['da_score_ss']
            # da_label_ss_resize = self._resize_sslb(da_score_ss)
            da_score_ss_resize = tf.reshape(da_score_ss, [-1, 2])
            da_label_ss_resize = self._resize_sslb(da_score_ss_resize) 
            da_conv_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=da_score_ss_resize,
                                                                   labels=da_label_ss_resize))
            # consisitency regularization loss
            resize_daConv_CR = self._resize_da_conv_score_for_CR(da_score_ss)
            daConv_minus_dc_ip3 = tf.subtract(resize_daConv_CR, dc_ip3)
            da_CR_loss = tf.nn.l2_loss(daConv_minus_dc_ip3)

            self._losses['dc_loss'] = dc_loss
            self._losses['da_conv_loss'] = da_conv_loss
            self._losses['da_CR_loss'] = da_CR_loss
            da_components_loss = dc_loss + da_conv_loss + da_CR_loss

            # --------------------DA Part End---------------------------

            loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box
            loss += 0.1 * da_components_loss
            regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')
            self._losses['total_loss'] = loss + regularization_loss

            self._event_summaries.update(self._losses)

        return loss

    def _resize_da_conv_score_for_CR(self, da_score_ss):
        # resize da_score_ss for consistency regularization compute loss
        # da_score_ss shape = (1, ?, ?, 2)
        # dc_ip3 shape = (256, 1)
        # need to resize da_score_ss to (1, 1) to compute loss with dc_ip3
        resize_daConv_CR = tf.reduce_mean(da_score_ss, (1, 2, 3))
        resize_daConv_CR = tf.reshape(resize_daConv_CR, (1,1))
        resize_daConv_CR = tf.tile(resize_daConv_CR, (256, 1))

        return resize_daConv_CR

    def _label_resize_layer(self, dc_ip3):
        # from caffe's label_resize_layer.py
        feats = dc_ip3
        lbs = self._dc_label
        
        feats_shape = tf.shape(feats)
        # lbs_tile = tf.tile(lbs, [feats_shape[0], 1])
        lbs_tile = tf.tile(lbs, [feats_shape[0], 1])

        # print("feats (type, shape[0]): ({}, {});".format(type(feats), feats.shape[0]))
        # print("feats: {}".format(feats))
        # 
        # print("lbs_tile (type, shape[0]): ({}, {});".format(type(lbs_tile), lbs_tile.shape[0]))
        # print("lbs_tile: {}".format(lbs_tile))

        lbs_resize = lbs_tile
        
        dc_label_resize = lbs_resize

        return dc_label_resize
        

    def _resize_sslb(self, da_score_ss_resize):
        # from caffe's resize_sslb.py
        feats = da_score_ss_resize  # [1900, 2]
        # da_label_ss_resize.reshape(1, 1, feats.shape[2], feats.shape[3])
        
        feats_shape = tf.shape(feats)
        lbs = self._need_backprop
        lbs_tile = tf.tile(lbs, [feats_shape[0], 1])  # reshape for image.resize([batch, row, col, channel]) 
        lbs_reshape = tf.reshape(lbs_tile, [-1])

        # resize image need to reverse the row and col
        # feats.shape[2(or3)] is Nonetype
        # feats_shape = feats.shape.as_list()
        # print("feats_shape:{}".format(feats_shape))
        # gt_blob = tf.image.resize(lbs, 
        #                           (feats.shape[1], feats.shape[2]),
        #                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # fit sparse__softmax_cross_entropy_with_logits
        # gt_blob = tf.reshape(gt_blob, (1, feats.shape[1], feats.shape[2]))

        # print("gt_blob (type, shape[0]): ({}, {});".format(type(gt_blob), gt_blob.shape))
        # print("gt_blob: {}".format(gt_blob))

        #gt_blob = tf.zeros((1, feats_shape[2], feats_shape[1], 1), dtype=np.float32)
        #gt_blob = lbs_resize
        #gt_blob = tf.reshape(gt_blob, [1, feats_shape[1], feats_shape[2], 1])
        
        # da_label_ss_resize.reshape(*gt_blob.shape)
        da_label_ss_resize = lbs_reshape

        return da_label_ss_resize

    def _region_proposal(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, 
                          weights_initializer=initializer,
                          scope="rpn_conv/3x3")
        self._act_summaries.append(rpn)
        rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_cls_score')

        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None, scope='rpn_bbox_pred')

        # change it so that the score has 2 as its channel size
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
        # shape : [1, low*col/2, channel, 2]
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
        # shape : [1, low*col/2, channel, 2]
        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
        # shape : [low*col/2*channel]
        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
        # shape : [1, low*col/2k, channel, 2k]

        if is_training:
            rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
            # Try to have a deterministic order for the computing graph, for reproducibility
            with tf.control_dependencies([rpn_labels]):
                rois, _ = self._proposal_target_layer(rois, roi_scores, "rpn_rois")
        else:
            if cfg.TEST.MODE == 'nms':
                rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            elif cfg.TEST.MODE == 'top':
                rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
            else:
                raise NotImplementedError

        self._predictions["rpn_cls_score"] = rpn_cls_score
        self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
        self._predictions["rpn_cls_prob"] = rpn_cls_prob
        self._predictions["rpn_cls_pred"] = rpn_cls_pred
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
        self._predictions["rois"] = rois
        return rois

    def _region_classification(self, fc7, is_training, initializer, initializer_bbox):
        cls_score = slim.fully_connected(fc7, self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training,
                                         activation_fn=None, scope='bbox_pred')

        self._predictions["cls_score"] = cls_score
        self._predictions["cls_pred"] = cls_pred
        self._predictions["cls_prob"] = cls_prob
        self._predictions["bbox_pred"] = bbox_pred

        return cls_prob, bbox_pred

    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def _tail_to_da(self, fc7, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_daConv(self, net_conv, is_training, resue=None):
        raise NotImplementedError

    def create_architecture(self, mode, num_classes, tag=None,
                            anchor_sizes=(128, 256, 512), anchor_strides=(16,), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._dc_label = tf.placeholder(tf.float32, shape=[1, 1])
        self._need_backprop = tf.placeholder(tf.int32, shape=[1, 1])
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode

        self._anchor_sizes = anchor_sizes
        self._num_sizes = len(anchor_sizes)
        self._anchor_strides = anchor_strides
        self._num_strides = len(anchor_strides)
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        self._num_anchors = self._num_sizes * self._num_ratios

        training = mode == 'TRAIN'
        testing = mode == 'TEST'

        assert tag != None

        # handle most of the regularizers here
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            rois, cls_prob, bbox_pred = self._build_network(training)

        layers_to_output = {'rois': rois}

        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        if testing:
            stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
            means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
            self._predictions["bbox_pred"] *= stds
            self._predictions["bbox_pred"] += means
        else:
            self._add_losses()
            layers_to_output.update(self._losses)

            val_summaries = []
            with tf.device("/cpu:0"):
                val_summaries.append(self._add_gt_image_summary())
                for key, var in self._event_summaries.items():
                    val_summaries.append(tf.summary.scalar(key, var))
                for key, var in self._score_summaries.items():
                    self._add_score_summary(key, var)
                for var in self._act_summaries:
                    self._add_act_summary(var)
                for var in self._train_summaries:
                    self._add_train_summary(var)

            self._summary_op = tf.summary.merge_all()
            self._summary_op_val = tf.summary.merge(val_summaries)

        layers_to_output.update(self._predictions)

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image,
                     self._im_info: im_info}

        cls_score, cls_prob, bbox_pred, rois = sess.run([self._predictions["cls_score"],
                                                         self._predictions['cls_prob'],
                                                         self._predictions['bbox_pred'],
                                                         self._predictions['rois']],
                                                        feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    def get_summary(self, sess, blobs):
        feed_dict = {self._image: blobs['data'],
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._dc_label: blobs['dc_label'],
                     self._need_backprop: blobs['need_backprop']}

        summary = sess.run(self._summary_op_val, feed_dict=feed_dict)

        return summary

    def train_step(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'],
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._dc_label: blobs['dc_label'],
                     self._need_backprop: blobs['need_backprop']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, dc_loss, da_conv_loss, da_CR_loss, loss, _ = \
                sess.run([self._losses["rpn_cross_entropy"],
                          self._losses['rpn_loss_box'],
                          self._losses['cross_entropy'],
                          self._losses['loss_box'],
                          self._losses['dc_loss'],
                          self._losses['da_conv_loss'],
                          self._losses['da_CR_loss'],
                          self._losses['total_loss'],
                          train_op],
                          feed_dict=feed_dict)

        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, dc_loss, da_conv_loss, da_CR_loss, loss

    def train_step_with_summary(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], 
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._dc_label: blobs['dc_label'],
                     self._need_backprop: blobs['need_backprop']}
        rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, dc_loss, da_conv_loss, da_CR_loss,loss, summary, _ = \
                sess.run([self._losses["rpn_cross_entropy"],
                          self._losses['rpn_loss_box'],
                          self._losses['cross_entropy'],
                          self._losses['loss_box'],
                          self._losses['dc_loss'],
                          self._losses['da_conv_loss'],
                          self._losses['da_CR_loss'],
                          self._losses['total_loss'],
                          self._summary_op,
                          train_op],
                          feed_dict=feed_dict)
        
        return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, dc_loss, da_conv_loss, da_CR_loss, loss, summary

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'],
                     self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes'],
                     self._dc_label: blobs['dc_label'],
                     self._need_backprop: blobs['need_backprop']}
        sess.run([train_op], feed_dict=feed_dict)
