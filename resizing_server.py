#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from flask import Flask, jsonify, request
import cv2
import numpy as np
import io

app = Flask(__name__)


# @app.route('/resize/<int:h>,<int:w>', methods=['POST'])
# def api_resize(h, w):
#     """
#     Input:
#         {'input': np.array:
#         'shape': tuple H,W}
#     Return :
#         {resized_image}
#     """
#     if request.is_json:
#         data = request.get_json()
#         resized_probs = cv2.resize(np.squeeze(np.array(data['input'], np.uint8)), tuple((w, h)))
#         return jsonify(resized_probs.tolist())
#     else:
#         raise TypeError("Request was not JSON")


@app.route('/resize/<int:h>,<int:w>', methods=['POST'])
def api_resize(h, w):
    """
    Input:
        {'input': np.array:
        'shape': tuple H,W}
    Return :
        {resized_image}
    """
    array = np.load(io.BytesIO(request.data))
    resized_array = cv2.resize(array, tuple((w, h)))

    buf = io.BytesIO()
    np.save(buf, resized_array)
    return buf.getvalue()


if __name__ == '__main__':
    app.run()
