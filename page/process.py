#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
from typing import List, Tuple
import numpy as np
from dh_segment.io import PAGE
from dh_segment.post_processing import binarization, boxes_detection


def page_binarization(probability_maps: dict,
                      threshold: float=-1,
                      kernel_size: int=15) -> np.array:
    """

    :param probability_maps:
    :param threshold:
    :param kernel_size:
    :return:
    """

    page_probs = probability_maps[:, :, 1]
    page_bin = binarization.thresholding(page_probs, threshold=threshold)
    page_bin = binarization.cleaning_binary(page_bin, kernel_size=kernel_size)

    return page_bin


def format_quad_to_string(quad):
    s = ''
    for corner in quad:
        s += '{},{},'.format(corner[0], corner[1])
    return s[:-1]


def process_probability_map_from_npy_file(npy_filename: str):
    """

    :param npy_filename:
    :return:
    """

    probs = np.load(npy_filename)
    basename = os.path.basename(npy_filename)

    page_bin = page_binarization(probs)

    # Set of 4 coordinates (x,y)
    page_coordinates = boxes_detection.find_boxes(page_bin, mode='min_rectangle', n_max_boxes=1)

    # # Create page region
    page_region = PAGE.Border(coords=[PAGE.Point(p[1], p[0]) for p in page_coordinates]) \
        if page_coordinates is not None else PAGE.Border()
    page = PAGE.Page(image_filename=basename.split('.')[0],
                     image_width=probs.shape[1],
                     image_height=probs.shape[0],
                     page_border=page_region)

    return page


def create_page_region_from_coordinates(filename: str,
                                        border_coordinates: np.array,
                                        image_shape: Tuple[int, int]) -> PAGE.Page:
    """

    :param filename:
    :param border_coordinates:
    :param image_shape:  (H, W)
    :return:
    """

    basename = os.path.basename(filename)

    page_region = PAGE.Border(coords=[PAGE.Point(p[1], p[0]) for p in border_coordinates]) \
        if border_coordinates is not None else PAGE.Border()
    page = PAGE.Page(image_filename=basename.split('.')[0],
                     image_width=image_shape[1],
                     image_height=image_shape[0],
                     page_border=page_region)

    return page


def get_page_box_from_probs(probabilty_map: np.array,
                            detection_mode: str='min_rectangle') -> np.array:

    page_bin = page_binarization(probabilty_map)

    # Set of 4 coordinates (x,y)
    return boxes_detection.find_boxes(page_bin, mode=detection_mode, n_max_boxes=1)


def process_probability_map_from_nparray(filename: str,
                                         probabilty_map: np.array) -> PAGE.Page:
    """

    :param filename:
    :param probabilty_map:
    :return:
    """

    page_coordinates = get_page_box_from_probs(probabilty_map)

    # # Create page region
    return create_page_region_from_coordinates(filename, page_coordinates, probabilty_map.shape[:2])


def export_page(page: PAGE.Page,
                output_page_dir: str,
                extension: str='json',
                comment='') -> None:
    """

    :param page:
    :param output_page_dir:
    :param extension:
    :param comment:
    :return:
    """

    basename_image = os.path.basename(page.image_filename)

    export_filename = os.path.join(output_page_dir, '{}.{}'.format(basename_image.split('.')[0], extension))
    page.write_to_file(export_filename,
                       creator_name='PageExtractor',
                       comments=comment)


def extract_page_from_probability_map(npy_filename: str,
                                      output_page_dir: str,
                                      model_dir: str) -> None:
    """

    :param npy_filename:
    :param output_page_dir:
    :param model_dir:
    :return:
    """

    page = process_probability_map_from_npy_file(npy_filename)
    export_page(page, output_page_dir, comment='| Page model : {} '.format(model_dir))

