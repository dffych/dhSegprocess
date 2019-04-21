#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from imageio import imread, imsave
import numpy as np
import cv2
import os
import pandas as pd
import re
from dh_segment.io import PAGE
from tqdm import tqdm
from glob import glob
from shutil import copyfile


def get_coords_form_txt_line(line: str)-> tuple:
    """
    gets the coordinates of the page from the txt file (line-wise)

    :param line: line of the .txt file
    :return: coordinates, filename
    """
    splits = line.split(',')
    full_filename = splits[0]
    splits = splits[1:]
    if splits[-1] in ['SINGLE', 'ABNORMAL']:
        coords_simple = np.reshape(np.array(splits[:-1], dtype=int), (4, 2))
        # coords_double = None
        coords = coords_simple
    else:
        coords_simple = np.reshape(np.array(splits[:8], dtype=int), (4, 2))
        # coords_double = np.reshape(np.array(splits[-4:], dtype=int), (2, 2))
        # coords = (coords_simple, coords_double)
        coords = coords_simple

    return coords, full_filename


def make_binary_mask(txt_file: str):
    """
    From export txt file with filnenames and coordinates of qudrilaterals, generate binary mask of page

    :param txt_file: txt file filename
    :return:
    """
    for line in open(txt_file, 'r'):
        dirname, _ = os.path.split(txt_file)
        c, full_name = get_coords_form_txt_line(line)
        img = imread(full_name)
        label_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        label_img = cv2.fillPoly(label_img, [c[:, None, :]], 255)
        basename = os.path.basename(full_name)
        imsave(os.path.join(dirname, '{}_bin.png'.format(basename.split('.')[0])), label_img)


def page_dataset_generator(txt_filename: str,
                           input_dir: str,
                           output_dir: str) -> None:
    """
    Given a txt file (filename, coords corners), generates a dataset of images + labels

    :param txt_filename: File (txt) containing list of images
    :param input_dir: Root directory to original images
    :param output_dir: Output directory for generated dataset
    :return:
    """

    output_img_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    for line in tqdm(open(txt_filename, 'r')):
        coords, full_filename = get_coords_form_txt_line(line)

        try:
            img = imread(os.path.join(input_dir, full_filename))
        except FileNotFoundError:
            print('File {} not found'.format(full_filename))
            continue
        label_img = np.zeros((img.shape[0], img.shape[1], 3))

        label_img = cv2.fillPoly(label_img, [coords], (255, 0, 0))
        # if coords_double is not None:
        #     label_img = cv2.polylines(label_img, [coords_double], False, color=(0, 0, 0), thickness=50)

        col, filename = full_filename.split(os.path.sep)[-2:]

        imsave(os.path.join(output_img_dir, '{}_{}.jpg'.format(col.split('_')[0], filename.split('.')[0])), img)
        imsave(os.path.join(output_label_dir, '{}_{}.png'.format(col.split('_')[0], filename.split('.')[0])), label_img)

    # Class file
    classes = np.stack([(0, 0, 0), (255, 0, 0)])
    np.savetxt(os.path.join(output_dir, 'classes.txt'), classes, fmt='%d')


def replace_variable_by_path_in_csv(csv_filename: str,
                                    export_csv_filename: str,
                                    root_image_dir: str,
                                    root_label_dir: str,
                                    image_dir_variable: str='\$ROOT_IMAGE_DIR',
                                    label_dir_variable: str='\$ROOT_LABEL_DIR') -> None:
    """
    Replace `image_dir_variable` and `label_dir_variable` by the desired path to image / label folders

    :param csv_filename:
    :param export_csv_filename:
    :param root_image_dir:
    :param root_label_dir:
    :param image_dir_variable:
    :param label_dir_variable:
    :return:
    """

    df_data = pd.read_csv(csv_filename, names=['image', 'label'])

    df_data['image'] = df_data['image'].apply(lambda x: re.sub(image_dir_variable,
                                                               root_image_dir,
                                                               x))

    df_data['label'] = df_data['label'].apply(lambda x: re.sub(label_dir_variable,
                                                               root_label_dir,
                                                               x))

    df_data.to_csv(export_csv_filename, header=False, index=False, encoding='utf8')


def gather_groundtruth_xml(root_dir: str,
                           export_dir: str) -> None:
    """
    From `root_dir` glob all the xml files in order to copy them into a single directory `export_dir`.

    :param root_dir:
    :param export_dir:
    :return:
    """

    os.makedirs(export_dir, exist_ok=True)

    xmls = glob(os.path.join(root_dir, '**', '**', 'page', '*.xml'))

    for filename in tqdm(xmls):
        splits = filename.split(os.path.sep)
        basename = '{}_{}'.format(splits[-3].split('_')[0], splits[-1])

        target_filename = os.path.join(export_dir, basename)

        copyfile(filename, target_filename)


def gather_groundtruth_json(annotation_file: str,
                            export_dir: str) -> None:
    """
    From the annotation txt file of PageNet repository, generate the json PAGE file with Page's Border

    :param annotation_file: txt file from PageNet
    :param export_dir: directory to export the json files
    :return:
    """

    with open(annotation_file, 'r', encoding='utf8') as f:
        content = f.read()

    lines = content.split('\n')

    for line in tqdm(lines):
        if line != '':
            splits = line.split(',')

            image_name = splits[0]
            filename_splits = image_name.split(os.path.sep)
            new_filename = '{}_{}.png'.format(filename_splits[1], filename_splits[-1].split('.')[0])

            coords = [int(c) for c in splits[1:9]]
            np_coords = np.stack([coords[::2], coords[1::2]], axis=1)

            page = PAGE.Page(image_filename=new_filename,
                             page_border=PAGE.Border(PAGE.Point.array_to_point(np_coords)))

            page.write_to_file(os.path.join(export_dir, '{}.json'.format(new_filename.split('.')[0])))


def format_image_dataset(root_images_dir: str, output_dir: str) -> None:
    """
    For a structure root_images_dir /<complex, simple>/<collection>/<img>.jpg, this takes all images ad puts
    them at the same directory level with filename formatting : <collection beginning>_<image filename>

    :param root_images_dir:
    :param output_dir:
    :return:
    """

    if os.path.isdir(output_dir):
        print("WARNING : Output directory '{}' already exist.".format(output_dir))
    else:
        os.makedirs(output_dir)

    filenames = glob(os.path.join(root_images_dir, '**', '**', '*.jpg')) + \
                glob(os.path.join(root_images_dir, '**', '*.jpg'))

    for filename in tqdm(filenames):
        splits = filename.split(os.path.sep)
        basename = splits[-1]
        collection = splits[-2].split('_')[0]
        new_filename = os.path.join(output_dir, '{}_{}'.format(collection, basename))

        copyfile(filename, new_filename)
