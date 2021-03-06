# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.throat_uvula import throat_uvula
from datasets.throat import throat
from datasets.polyp import polyp
from datasets.coco import coco
from datasets.cell import cell
import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}_diff'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year,
                                                                  use_diff=True))
# Set up voc_<year>_<split> 
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year,
                                                                  use_diff=False,
                                                                  extra_string='_frege'))
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'voc_{}_{}_diff_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year,
                                                                  use_diff=True,
                                                                  extra_string='_frege'))
# Set up throat_uvula_<year>_<split>
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'throat_uvula_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: throat_uvula(split, year))
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'throat_uvula_{}_{}_diff'.format(year, split)
        __sets[name] = (lambda split=split, year=year: throat_uvula(split, year,use_diff=True))
# Set up throat_uvula_<year>_<split>
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'throat_uvula_{}_{}_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: throat_uvula(split, year,use_diff=False,extra_string='_frege'))
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'throat_uvula_{}_{}_diff_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: throat_uvula(split, year,use_diff=True,extra_string='_frege'))


# Set up throat_<year>_<split> 
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'throat_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: throat(split, year))
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'throat_{}_{}_diff'.format(year, split)
        __sets[name] = (lambda split=split, year=year: throat(split, year,use_diff=True))
# Set up throat_<year>_<split> 
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'throat_{}_{}_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: throat(split, year,use_diff=False,extra_string='_frege'))
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'throat_{}_{}_diff_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: throat(split, year,use_diff=True,extra_string='_frege'))


# Set up polyp_<year>_<split>
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'polyp_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: polyp(split, year))
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'polyp_{}_{}_diff'.format(year, split)
        __sets[name] = (lambda split=split, year=year: polyp(split, year,use_diff=True))
# Set up polyp_<year>_<split>
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'polyp_{}_{}_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: polyp(split, year,use_diff=False,extra_string='_frege'))
for year in ['2007', '2012', '2012_test']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = 'polyp_{}_{}_diff_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: polyp(split, year,use_diff=True,extra_string='_frege'))


# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year))

# Set up coco_2014_<split>
for year in ['2014']:
    for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
        name = 'coco_{}_{}_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year,
                                                            extra_string='_frege'))

# Set up coco_2015_<split>
for year in ['2015']:
    for split in ['test', 'test-dev']:
        name = 'coco_{}_{}_frege'.format(year, split)
        __sets[name] = (lambda split=split, year=year: coco(split, year,
                                                            extra_string='_frege'))

for split in ['train', 'val']:
    name = 'cell_{}'.format(split)
    __sets[name] = (lambda split=split: cell(split))


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
