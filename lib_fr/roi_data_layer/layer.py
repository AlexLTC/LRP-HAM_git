# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import time


class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes, random=False):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        # Also set a random flag
        self._random = random
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        # If the random flag is set,
        # then the database is shuffled according to system time
        # Useful for the validation set
        if self._random:
            st0 = np.random.get_state()
            millis = int(round(time.time() * 1000)) % 4294967295
            np.random.seed(millis)

        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            if not self._random:  # train mode
                print("------------------------------------Show RoIDataLayer Check Result-------------------------------")
                # divide source & target from roidb
                self._source_inds = [i for i in range(len(self._roidb)) if self._roidb[i]['image'].find('source') != -1]
                self._target_inds = [i for i in range(len(self._roidb)) if self._roidb[i]['image'].find('target') != -1]
                # print("source index len: {}".format(len(self._source_inds)))
                print("source index: {}".format(self._source_inds[:5]))
                # print("target index len: {}".format(len(self._target_inds)))
                print("target index: {}".format(self._target_inds[:5]))

                assert len(self._source_inds) == 24314, 'source index length error in layer.py'
                assert len(self._target_inds) == 1128, 'target index length error in layer.py'
                

                # shuffle source & target index independently
                source_perm = np.random.permutation(self._source_inds)
                target_perm = np.random.permutation(self._target_inds)
                while(len(source_perm)/len(target_perm) > 1):
                    tmp_perm = np.random.permutation(self._target_inds)
                    target_perm = np.concatenate((target_perm, tmp_perm), 0)

                target_perm = target_perm[:len(source_perm)]
                print("source perm: {}".format(source_perm[:5]))
                print("target perm: {}".format(target_perm[:5]))

                # concate source & target index (length depends on source dataset)
                # if reach target data max number (shuffle target data then concate)
                perm_concate = []
                for i in range(len(source_perm)):
                    perm_concate.append(source_perm[i])
                    perm_concate.append(target_perm[i])

                print("perm concate length: {}".format(len(perm_concate)))
                print("perm concate: {}".format(perm_concate[:5]))

                assert len(perm_concate) == 48628, "perm concate length error in layer.py"

                self._perm = perm_concate
            else:  # val mode    
                self._perm = np.random.permutation(np.arange(len(self._roidb)))
        # Restore the random state
        if self._random:
            np.random.set_state(st0)

        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        # if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._perm):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs
