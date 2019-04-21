#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
from glob import glob
import numpy as np
from imageio import imread
from tqdm import tqdm
from typing import List
from process import process_probability_map_from_npy_file
from dh_segment.utils.evaluation import intersection_over_union, Metrics
from dh_segment.post_processing import boxes_detection
from dh_segment.io import PAGE
from shapely.geometry import Polygon

# PP_PARAMS = {'threshold': -1, 'kernel_size': 5}
#
#
# def eval_fn(input_dir: str,
#             groundtruth_dir: str) -> Metrics:
#     """
#
#     :param input_dir: directory containing the predictions .npy files (range [0, 255])
#     :param groundtruth_dir: directory containing the ground truth images (.png) (must have the same name as predictions
#                             files in input_dir)
#     :param post_process_params: params for post processing fn
#     :return: Metrics object containing all the necessary metrics
#     """
#     global_metrics = Metrics()
#     for file in tqdm(glob(os.path.join(input_dir, '*.npy'))):
#         basename = os.path.basename(file).split('.')[0]
#
#         label_image = imread(os.path.join(groundtruth_dir, '{}.png'.format(basename)), pilmode='L')
#
#         pred_box = process_probability_map_from_npy_file(file)
#         label_box = boxes_detection.find_boxes(label_image / np.max(label_image), min_area=0.0)
#
#         if pred_box is not None and label_box is not None:
#             iou = intersection_over_union(label_box.resize(-1, 1, 2),
#                                           pred_box.resize(-1, 1, 2),
#                                           label_image.shape)
#             global_metrics.IOU_list.append(iou)
#         else:
#             global_metrics.IOU_list.append(0)
#
#     global_metrics.compute_miou()
#     print('EVAL --- mIOU : {}\n'.format(global_metrics.mIOU))
#
#     return global_metrics


def eval_miou(label_filenames: List[str],
              prediction_json_dir: str,
              groundtruth_json_dir: str):
    """

    :param label_filenames:
    :param prediction_json_dir:
    :param groundtruth_json_dir:
    :return:
    """

    # Take basename and remove extension from filename
    basenames_labels = map(lambda x: os.path.basename(x).split('.')[0], label_filenames)

    list_iou = list()

    for elem in tqdm(basenames_labels):

        prediction_json = os.path.join(prediction_json_dir, elem + '.json')
        gt_json = os.path.join(groundtruth_json_dir, elem + '.json')
        assert os.path.isfile(prediction_json)
        assert os.path.isfile(gt_json)

        # Get page polygon for prediction
        page_prediction = PAGE.parse_file(prediction_json)
        border_prediction = PAGE.Point.point_to_list(page_prediction.page_border.coords)
        polygon_prediction = Polygon(border_prediction)

        # Get page polygon for groundtruth
        page_gt = PAGE.parse_file(gt_json)
        border_gt = PAGE.Point.point_to_list(page_gt.page_border.coords)
        polygon_gt = Polygon(border_gt)

        iou = polygon_prediction.intersection(polygon_gt).area / polygon_prediction.union(polygon_gt).area

        list_iou.append(iou)

    m_iou = np.mean(list_iou)
    print(m_iou)

    return list_iou
