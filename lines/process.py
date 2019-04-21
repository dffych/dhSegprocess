#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from dh_segment.post_processing import binarization
import numpy as np
import cv2
import os
import re
from uuid import uuid4
from typing import List, Tuple
from shapely.geometry import MultiLineString, Polygon
from dh_segment.io import PAGE
from dh_segment.post_processing import line_vectorization

OUTPUT_PAGE_DIR = 'page'


def _vertical_local_maxima(probability_map: np.array) -> np.array:
    """

    :param probability_map:
    :return:
    """
    local_maxima = np.zeros_like(probability_map, dtype=bool)
    local_maxima[1:-1] = (probability_map[1:-1] >= probability_map[:-2]) & (probability_map[2:] <= probability_map[1:-1])
    local_maxima = cv2.morphologyEx(local_maxima.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    return local_maxima > 0


def _mid_mean_median_threshold(probability_map: np.array,
                               min_prob: float=0.3) -> float:
    valid_values = probability_map[probability_map > min_prob]
    return 0.5*np.median(valid_values) + 0.5*np.mean(valid_values)


def _build_output_folder(output_root_dir: str, filename_image: str):
    """

    :param output_root_dir:
    :param filename_image:
    :return:
    """
    if not output_root_dir:
        return os.path.join(os.path.dirname(filename_image), OUTPUT_PAGE_DIR)
    else:
        # Find the uuid folder name and subfolders
        pattern = re.search("[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
                            filename_image)
        if pattern:
            return os.path.join(output_root_dir, os.path.dirname(filename_image[pattern.start():]), OUTPUT_PAGE_DIR)
        else:
            return os.path.join(output_root_dir)


def baselines_binarization(probability_map: np.ndarray,
                           vertical_maxima: bool=False,
                           low: float=0.3,
                           high: float=0.6) -> np.ndarray:
    """
    Given an image of probabilities, binarizes it to obtain a mask of lines.

    :param probability_map: np.array HxW in [0,1]
    :param vertical_maxima: flag to enable the preselection of line candidates, based on vertical local maxima
    :param low: low threshold value for hysteresis
    :param high: high threshold value for hysteresis
    :return: binary image HxW with lines
    """
    clean_probs = binarization.cleaning_probs(probability_map, sigma=1.5)
    mask_baselines = binarization.hysteresis_thresholding(clean_probs, low_threshold=low, high_threshold=high,
                                                          candidates_mask=_vertical_local_maxima(clean_probs) if
                                                          vertical_maxima else None)

    return mask_baselines


def lines_binarization(probability_map: np.ndarray,
                       threshold: float=-1) -> np.ndarray:
    """

    :param probability_map:
    :param threshold:
    :return:
    """

    if np.median(probability_map[probability_map > 0.2]) > 0.6 and threshold < 0:
        threshold = _mid_mean_median_threshold(probability_map, 0.2)

    mask_lines = binarization.thresholding(probability_map, threshold=threshold)
    mask_lines = binarization.cleaning_binary(mask_lines)
    return mask_lines


def endpoint_binarization(probability_map: np.array):
    pass
    # binarize endpoint

    # skeltonize

    # dilate by 2-3 pixels

    # return


def _orient_baseline(baseline: np.array):
    """

    :param baseline: (N,2) array
    :return:
    """
    if baseline[0, 0] > baseline[-1, 0]:
        return baseline[::-1]
    else:
        return baseline


def assign_baseline_to_lines(baselines_coordinates: List[np.array],
                             line_contours_coordinates: List[np.array]) -> List[PAGE.TextLine]:
    """

    :param baselines_coordinates: list of array of shape (N,2)
    :param line_contours_coordinates:
    :return:
    """
    baselines_coordinates = [_orient_baseline(bl) for bl in baselines_coordinates]

    baselines_set = MultiLineString([bl.reshape((bl.shape[0], bl.shape[1]))
                                     for bl in baselines_coordinates if len(bl) > 1])

    if len(baselines_set) > 1:
        baselines_set = baselines_set.simplify(1, preserve_topology=True)
    # else:
    #     print(baselines_set)

    polygons_set = [Polygon(line.reshape((line.shape[0], line.shape[1])))
                    for line in line_contours_coordinates if len(line) > 2]
    # polygons_set = [polygon.simplify(1, preserve_topology=False) for polygon in polygons_set]
    # TODO : we could simplify polylines and baselines with geometry.simplify

    # Remove invalid polygons
    number_found_polygons = len(polygons_set)
    polygons_set = [p for p in polygons_set if p.is_valid]
    if len(polygons_set) != number_found_polygons:
        print('There were {} non-valid polygon(s) removed'
              .format(number_found_polygons - len(polygons_set)))

    # For each polygon, lists the baseline indexes that intersects it
    polygon_intersections = [
        [i_b for i_b, baseline in enumerate(baselines_set) if polygon.intersects(baseline)]
        for polygon in polygons_set]
    # For each baseline, lists the polygons indexes that intersect it
    baselines_intersections = [
        [i_p for i_p, polygon in enumerate(polygons_set) if baseline.intersects(polygon)]
        for baseline in baselines_set]

    page_textlines = list()
    already_processed_baseline_intersections = list()
    for p_index, b_index_list in enumerate(polygon_intersections):

        if len(b_index_list) == 1:  # case : one polyline, one baseline
            b_index = b_index_list[0]

            if len(baselines_intersections[b_index]) == 1:  # If the intersection is mutually exclusive
                # Create a PAGE textline with line coords = polygon coords and baseline points
                page_textlines.append(
                    PAGE.TextLine(
                        id=str(uuid4()),
                        coords=PAGE.Point.array_to_point(np.array(polygons_set[p_index].exterior.coords, dtype=int)),
                        baseline=PAGE.Point.array_to_point(np.array(baselines_set[b_index].coords, dtype=int))
                    )
                )

            else:  # case with one baseline and multiple polylines -> split
                # in order to avoid to process multiple times the same polygons :
                if baselines_intersections[b_index] in already_processed_baseline_intersections: continue
                already_processed_baseline_intersections.append(baselines_intersections[b_index])

                for intersected_polygon_index in baselines_intersections[b_index]:
                    split_baseline_coords = list()
                    for coords in baselines_set[b_index].coords:
                        if polygons_set[intersected_polygon_index].bounds[0] <= coords[0] \
                                <= polygons_set[intersected_polygon_index].bounds[2]:
                            split_baseline_coords.append(coords)

                    page_textlines.append(
                        PAGE.TextLine(
                            id=str(uuid4()),
                            coords=PAGE.Point.array_to_point(
                                np.array(polygons_set[intersected_polygon_index].exterior.coords, dtype=int)),
                            baseline=PAGE.Point.array_to_point(np.array(split_baseline_coords, dtype=int))
                        )
                    )
        elif len(b_index_list) > 1:  # case with one polyline and multiple baselines
            # create one textline per baseline
            for b_index in b_index_list:
                page_textlines.append(
                    PAGE.TextLine(
                        id=str(uuid4()),
                        coords=PAGE.Point.array_to_point(np.array(polygons_set[p_index].exterior.coords, dtype=int)),
                        baseline=PAGE.Point.array_to_point(np.array(baselines_set[b_index].coords, dtype=int))
                    )
                )

            # baseline_coords = [c for b_index in b_index_list for c in baselines_set[b_index].coords]
            # baseline_coords.sort(key=lambda c: c[0])  # sort from xmin to xmax
            #
            # page_textlines.append(
            #     PAGE.TextLine(
            #         id=str(uuid4()),
            #         coords=PAGE.Point.array_to_point(np.array(polygons_set[p_index].exterior.coords, dtype=int)),
            #         baseline=PAGE.Point.array_to_point(np.array(baseline_coords, dtype=int))
            #     )
            # )
        else:  # No intersection with baseline
            page_textlines.append(
                PAGE.TextLine(
                    id=str(uuid4()),
                    coords=PAGE.Point.array_to_point(np.array(polygons_set[p_index].exterior.coords, dtype=int))
                )
            )

    # Sort TextLines, lowest y coord first
    page_textlines.sort(key=lambda line: np.mean([c.y for c in line.coords]))

    return page_textlines


def generate_text_lines_regions_from_lines(lines_coordinates: List[np.array]):
    """

    :param lines_coordinates: List of (N, 2) coordinates
    :return:
    """
    page_textlines = list()

    for coordinates in lines_coordinates:
        page_textlines.append(PAGE.TextLine(id=str(uuid4()),
                                            coords=PAGE.Point.array_to_point(np.array(coordinates, dtype=int))))

    # Sort TextLines, lowest y coord first
    page_textlines.sort(key=lambda line: np.mean([c.y for c in line.coords]))

    return page_textlines


def generate_text_lines_regions_from_baselines(baseline_coordinates: List[np.array]):
    """

    :param baseline_coordinates: List of (N, 2) coordinates
    :return:
    """
    page_textlines = list()

    for coordinates in baseline_coordinates:
        page_textlines.append(PAGE.TextLine(id=str(uuid4()),
                                            baseline=PAGE.Point.array_to_point(np.array(coordinates, dtype=int))))

    # Sort TextLines, lowest y coord first
    page_textlines.sort(key=lambda line: np.mean([c.y for c in line.baseline]))

    return page_textlines


def vectorization(probability_maps: np.array) -> (List[np.array], List[np.array]):
    """

    :param probability_maps:
    :return: tuple baselines coordinates, contours lines in opencv format list of (N,1,2) points
    """

    # Todo : vectorize / mask endpoints

    mask_baselines = baselines_binarization(probability_maps[:, :, 0], vertical_maxima=False, low=0.4, high=0.7)
    mask_lines = lines_binarization(probability_maps[:, :, 2])

    baselines = line_vectorization.find_lines(mask_baselines)
    _, contours_lines, _ = cv2.findContours(mask_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return baselines, contours_lines


def _create_page(image_filename: str,
                 list_text_lines: List[PAGE.TextLine],
                 original_shape: tuple) -> PAGE.Page:
    """

    :param image_filename:
    :param list_text_lines:
    :param original_shape:
    :return:
    """
    border_coords = [[0, 0],
                     [original_shape[1], 0],
                     [original_shape[1], original_shape[0]],
                     [0, original_shape[0]]]
    page_border = PAGE.Border(coords=[PAGE.Point(p[1], p[0]) for p in border_coords])
    page = PAGE.Page(image_filename=image_filename, image_width=int(original_shape[1]),
                     image_height=int(original_shape[0]), page_border=page_border)

    # Create a text region
    text_region = PAGE.TextRegion(id='page',
                                  coords=page.page_border.coords,
                                  text_lines=list_text_lines)

    page.text_regions = [text_region]

    return page


def create_page_region_with_lines(image_filename: str,
                                  list_text_lines: List[PAGE.TextLine],
                                  original_shape: tuple,
                                  output_dir: str,
                                  file_extension: str='json',
                                  comments: str=None):

    """

    :param image_filename:
    :param list_text_lines:
    :param original_shape:
    :param output_dir:
    :param file_extension: either 'xml' or 'json'
    :param comments:
    :return:
    """

    assert file_extension in ['json', 'xml']

    basename = os.path.basename(image_filename).split('.')[0]

    # Build output dir path
    pages_dir = _build_output_folder(output_dir, image_filename)
    if not os.path.exists(pages_dir):
        os.makedirs(pages_dir)
    page_filename = os.path.join(pages_dir, '{}.{}'.format(basename, file_extension))
    if os.path.exists(page_filename):
        page = PAGE.parse_file(page_filename)
    else:
        page = _create_page(image_filename, list_text_lines, original_shape)

    page.write_to_file(page_filename,
                       creator_name='LinesExtractor',
                       comments=comments)


def page_jsonify(image_filename: str,
                 list_text_lines: List[PAGE.TextLine],
                 original_shape: tuple):

    """

    :param image_filename:
    :param list_text_lines:
    :param original_shape:
    :return:
    """

    page = _create_page(image_filename, list_text_lines, original_shape)

    return page.to_json()
