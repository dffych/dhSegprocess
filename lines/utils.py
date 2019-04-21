#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import os
import re
import pandas as pd


def rename_image_filename(image_filename: str) -> str:
    dirname = os.path.dirname(image_filename)
    basename = os.path.basename(image_filename).split('.')[0]

    # splits = dirname.split(os.sep)
    # root_prefix = splits[0]
    basename_prefix = dirname.split(os.sep)[-1].split('_')[0]
    new_basename = '{}_{}.jpg'.format(basename_prefix, basename)
    return os.path.join('$ROOT_IMAGE_DIR', new_basename)


def rename_label_filename(label_filename: str) -> str:
    basename = os.path.basename(label_filename)
    return os.path.join('$ROOT_LABEL_DIR', basename)


def replace_variable_by_path_in_csv(csv_filename: str,
                                    output_csv_filename: str,
                                    root_image_dir: str,
                                    root_label_dir: str,
                                    image_dir_variable: str = '\$ROOT_IMAGE_DIR',
                                    label_dir_variable: str = '\$ROOT_LABEL_DIR'):

    df = pd.read_csv(csv_filename, header=None, names=['image', 'label'])

    df['image'] = df.image.apply(lambda x: re.sub(image_dir_variable, root_image_dir, x))
    df['label'] = df.label.apply(lambda x: re.sub(label_dir_variable, root_label_dir, x))

    assert (sum(df.image.apply(lambda x: not os.path.isfile(x))) == 0)
    assert (sum(df.label.apply(lambda x: not os.path.isfile(x))) == 0)

    df.to_csv(output_csv_filename, header=False, index=False, encoding='utf8')
