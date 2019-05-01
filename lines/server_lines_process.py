#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from flask import Flask, request, jsonify
import cv2
import numpy as np
import io
import json
from process import vectorization, create_page_region_with_lines, assign_baseline_to_lines,\
    generate_text_lines_regions_from_baselines, generate_text_lines_regions_from_lines, page_jsonify
from taputapu.transform.affine import resize_image_coordinates

app = Flask(__name__)


@app.route('/resize/<int:h>,<int:w>', methods=['POST'])
def api_resize(h, w):
    """
    Input:
        {'probs': np.array}
    Return :
        {resized iamge}
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
        {'probs': np.array
        'was_resized': bool,
        'original_shape'; tuple H,W}
    Output:
        {'baselines': list of baselines,
        'lines': list of lines}
    """

    probs = np.load(io.BytesIO(request.data))
    if int(was_resized) > 0:
        baselines, contours_lines = vectorization(probs)

        baselines = [bl[:, 0, :].tolist() for bl in baselines]
        contours_lines = [line[:, 0, :].tolist() for line in contours_lines]

        return jsonify(baselines=baselines,
                       lines=contours_lines)

    else:
        baselines, contours_lines = vectorization(probs)

        # Upscale coordinates
        baselines_resized = [resize_image_coordinates(bl[:, 0, :], probs.shape[:2], (h, w)).tolist() for bl in
                             baselines]
        lines_resized = [resize_image_coordinates(line[:, 0, :], probs.shape[:2], (h, w)).tolist() for line in
                         contours_lines]

        return jsonify(baselines=baselines_resized,
                       lines=lines_resized)


@app.route('/export-regions/<string:filename_image>', methods=['POST'])
def api_export_regions(filename_image):
    """
    Input:
        {'baselines':
        'lines':
        'shape_image':}
    :return:
    """
    if request.is_json:
        data = request.get_json()
        baselines = [np.array(baseline) for baseline in data['baselines']]
        contours_lines = [np.array(line) for line in data['lines']]

        if baselines is not None and contours_lines is not None:
            list_text_lines = assign_baseline_to_lines(baselines, contours_lines)
        elif baselines is None and contours_lines is not None:
            list_text_lines = generate_text_lines_regions_from_lines(contours_lines)
        elif baselines is not None and contours_lines is None:
            list_text_lines = generate_text_lines_regions_from_baselines(baselines)
        else:
            list_text_lines = list()

        return json.dumps(page_jsonify(image_filename=filename_image,
                                       list_text_lines=list_text_lines,
                                       original_shape=data['shape_image']))

    else:
        raise TypeError("Request was not JSON")


@app.route('/process/<string:filename_image>/<int:h>,<int:w>/<string:resizing>', methods=['POST'])
def api_process(filename_image: str, h: int, w: int, resizing: bool):

    data = request.data
    probs = np.load(io.BytesIO(data))

    if probs.shape[0] == 1:
        probs = probs[0, :, :, :]

    if resizing in ['true', 'TRUE', 'True']:
        resized_probs = cv2.resize(np.squeeze(probs), tuple((w, h)))

        baselines, contours_lines = vectorization(resized_probs)

        baselines = [bl[:, 0, :].tolist() for bl in baselines]
        contours_lines = [line[:, 0, :].tolist() for line in contours_lines]

    elif resizing in ['false', 'FALSE', 'False']:
        baselines, contours_lines = vectorization(probs)

        # Upscale coordinates
        baselines = [resize_image_coordinates(bl[:, 0, :], probs.shape[:2], (h, w)).tolist() for bl in baselines]
        contours_lines = [resize_image_coordinates(line[:, 0, :], probs.shape[:2], (h, w)).tolist() for line in
                          contours_lines]

    else:
        raise NotImplementedError('The is no option {} availbale for resizing parameter '
                                  '(use "true" or "false")'.format(resizing))

    baselines = [np.array(baseline) for baseline in baselines]
    contours_lines = [np.array(line) for line in contours_lines]

    if baselines is not None and contours_lines is not None:
        list_text_lines = assign_baseline_to_lines(baselines, contours_lines)
    elif baselines is None and contours_lines is not None:
        list_text_lines = generate_text_lines_regions_from_lines(contours_lines)
    elif baselines is not None and contours_lines is None:
        list_text_lines = generate_text_lines_regions_from_baselines(baselines)
    else:
        list_text_lines = list()

    return json.dumps(page_jsonify(image_filename=filename_image,
                                   list_text_lines=list_text_lines,
                                   original_shape=(h, w)))


@app.route('/processfile', methods=['POST'])
def api_process():

    data = request.data
    filename_image = data.filename
    h = data.h
    w = data.w
    resizing = data.resizing
    
    with open(filename_image, 'r') as infile:
        data = json.load(infile)

    buf = io.BytesIO()
    np.save(buf, data["outputs"]["probs"])
    probs = buf.getvalue()
    
    #probs = np.load(io.BytesIO(data))

    if probs.shape[0] == 1:
        probs = probs[0, :, :, :]

    if resizing in ['true', 'TRUE', 'True']:
        resized_probs = cv2.resize(np.squeeze(probs), tuple((w, h)))

        baselines, contours_lines = vectorization(resized_probs)

        baselines = [bl[:, 0, :].tolist() for bl in baselines]
        contours_lines = [line[:, 0, :].tolist() for line in contours_lines]

    elif resizing in ['false', 'FALSE', 'False']:
        baselines, contours_lines = vectorization(probs)

        # Upscale coordinates
        baselines = [resize_image_coordinates(bl[:, 0, :], probs.shape[:2], (h, w)).tolist() for bl in baselines]
        contours_lines = [resize_image_coordinates(line[:, 0, :], probs.shape[:2], (h, w)).tolist() for line in
                          contours_lines]

    else:
        raise NotImplementedError('The is no option {} availbale for resizing parameter '
                                  '(use "true" or "false")'.format(resizing))

    baselines = [np.array(baseline) for baseline in baselines]
    contours_lines = [np.array(line) for line in contours_lines]

    if baselines is not None and contours_lines is not None:
        list_text_lines = assign_baseline_to_lines(baselines, contours_lines)
    elif baselines is None and contours_lines is not None:
        list_text_lines = generate_text_lines_regions_from_lines(contours_lines)
    elif baselines is not None and contours_lines is None:
        list_text_lines = generate_text_lines_regions_from_baselines(baselines)
    else:
        list_text_lines = list()

    return json.dumps(page_jsonify(image_filename=filename_image,
                                   list_text_lines=list_text_lines,
                                   original_shape=(h, w)),filename_image+".lines")


if __name__ == '__main__':
    app.run(host='0.0.0.0')
