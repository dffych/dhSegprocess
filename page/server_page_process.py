#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
import json
from process import get_page_box_from_probs, create_page_region_from_coordinates
from taputapu.transform.affine import resize_image_coordinates

app = Flask(__name__)


@app.route('/resize/<int:h>,<int:w>', methods=['POST'])
def api_resize(h: int, w: int):
    """
    Input:
        BytesIO content : image to resize
    Return :
        ByteIO content : image resized
    """
    data = request.data
    array = np.load(io.BytesIO(data))

    resized_probs = cv2.resize(np.squeeze(array), tuple((w, h)))

    buf = io.BytesIO()
    np.save(buf, resized_probs)
    return buf.getvalue()


@app.route('/post-process/<was_resized>/<int:h>,<int:w>', methods=['POST'])
def api_post_process(was_resized: bool, h: int, w: int):
    """
    post processes the probability maps and returns the coordinates of lines and baselines
    Input :
        BytesIO content :
    Output:
        {'page_coordinates': coordinates of page (4 corners)}
    """

    probs = np.load(io.BytesIO(request.data))
    if int(was_resized) > 0:
        page_coordinates = get_page_box_from_probs(probs)

        return jsonify(page_coordinates=page_coordinates.tolist())

    else:
        page_coordinates = get_page_box_from_probs(probs)
        page_coordinates_resized = resize_image_coordinates(page_coordinates, probs.shape[:2], (h, w))

        return jsonify(page_coordinates=page_coordinates_resized.tolist())


@app.route('/export-regions/<string:filename_image>', methods=['POST'])
def api_export_regions(filename_image):
    """
    Input :
        {'page_coordinates':
        'shape_image': }
    :param filename_image:
    :return:
    """
    if request.is_json:
        data = request.get_json()
        page = create_page_region_from_coordinates(filename_image, data['page_coordinates'], data['shape_image'])
        return json.dumps(page.to_json())

    else:
        raise TypeError("Request was not JSON")


@app.route('/process/<string:filename_image>/<int:h>,<int:w>/<string:resizing>', methods=['POST'])
def api_process(filename_image: str, h: int, w: int, resizing: bool=True):

    data = request.data
    probs = np.load(io.BytesIO(data))

    if probs.shape[0] == 1:
        probs = probs[0, :, :, :]

    if resizing in ['true', 'TRUE', 'True']:
        resized_probs = cv2.resize(probs, tuple((w, h)))
        page_coordinates = get_page_box_from_probs(resized_probs)

    elif resizing in ['false', 'FALSE', 'False']:
        page_coordinates = get_page_box_from_probs(probs)
        page_coordinates = resize_image_coordinates(page_coordinates, probs.shape[:2], (h, w))
    else:
        raise NotImplementedError('The is no option {} availbale for resizing parameter '
                                  '(use "true" or "false")'.format(resizing))

    page = create_page_region_from_coordinates(filename_image, page_coordinates, (h,w))

    return json.dumps(page.to_json())


if __name__ == '__main__':
    app.run(host='0.0.0.0')
