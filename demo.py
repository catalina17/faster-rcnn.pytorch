# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np

from model.faster_rcnn.vgg16 import vgg16


def get_frcnn_feature_extractor():
  pascal_classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
  fasterRCNN = vgg16(pascal_classes, pretrained=True)
  fasterRCNN.create_architecture()
  print('Loaded pre-trained Faster R-CNN model successfully!')
  print(fasterRCNN.RCNN_base[:10])

  fasterRCNN.eval()
  return fasterRCNN.RCNN_base[:10]
