#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

from typing import List, Tuple
from multiprocessing import JoinableQueue, Queue, Pool, Event
from multiprocessing.pool import ThreadPool
import tensorflow as tf
import numpy as np
import cv2
from time import time
from tqdm import tqdm
from dh_segment.inference import LoadedModel, BatchLoadedModel
from itertools import islice


_MAX_SLEPT_TIME = 10


def _resize(filename: str,
            probs: np.ndarray,
            original_shape: np.array,
            queue: Queue):
    """
    Resize (upscale) probability maps.

    :param filename:
    :param probs:
    :param original_shape:
    :param queue: queue to put resized data
    :return:
    """

    original_shape = original_shape
    resized_probs = cv2.resize(probs, tuple(original_shape[::-1]))

    queue.put((filename, resized_probs))
    # print('+ put data in pp-queue')


def _crop(probability_map: np.ndarray,
          resized_shape: np.array) -> np.ndarray:
    """
    Removes the right/bottom padding added for batch processing

    :param probability_map:
    :param resized_shape:
    :return:
    """
    h, w = resized_shape
    return probability_map[:h, :w, :]


def _crop_and_resize(filename: str,
                     probability_maps: np.array,
                     resized_shape: Tuple[int, int],
                     original_shape: Tuple[int, int],
                     post_processing_queue: JoinableQueue) -> None:
    """

    :param filename:
    :param probability_maps:
    :param resized_shape:
    :param original_shape:
    :param post_processing_queue:
    :return:
    """

    cropped_probabilities = _crop(probability_maps, resized_shape)
    _resize(filename, cropped_probabilities, original_shape, post_processing_queue)


def prediction_simple(filenames_to_predict: List[str],
                      model_dir: LoadedModel,
                      queue: Queue):
    """
    GPU prediction with single filename

    :param filenames_to_predict:
    :param model_dir:
    :param queue: queue to enqueue tuple (filename, prediction)
    :return:
    """

    with tf.Session():
        model = LoadedModel(model_dir)

        for image_filename in tqdm(filenames_to_predict):
            predictions = model.predict(image_filename)

            queue.put((image_filename, predictions))
            # print('+ put data in gpu_quque')


def prediction_batch(filenames_to_predict: List[str],
                     model_dir: str,
                     queue: JoinableQueue,
                     batch_size: int=8,
                     config: tf.ConfigProto=tf.ConfigProto()) -> None:
    """
    GPU predition with batches

    :param filenames_to_predict:
    :param model_dir:
    :param batch_size:
    :param queue: queue to enqueue tuple (filename, predictions)
    :param config:
    :return:
    """

    iterator_filenames = iter(filenames_to_predict)

    with tf.Session(config=config):
        m = BatchLoadedModel(model_dir)
        m.init_prediction(filenames_to_predict, batch_size)

        it = 0
        next_batch = list(islice(iterator_filenames, batch_size))
        while True:
            try:
                predictions = m.predict_next_batch()

                queue.put((next_batch, predictions))

                next_batch = list(islice(iterator_filenames, batch_size))
                print('Predicted {} / {}'.format((it + 1) * batch_size, len(filenames_to_predict)))
                it += 1

            except tf.errors.OutOfRangeError:
                # Block until all items in the queue have been gotten and processed.
                # queue.join()
                return


# ----------------------------------------------------

def _wk_crop_and_resize_from_queue(queue_from_gpu: JoinableQueue,
                                   queue_to_post_process: JoinableQueue,
                                   n_processes: int=None):
    """
    Worker for croping and resizing elements of the GPU queue. Use only when batch prediction is used

    :param queue_from_gpu: queue to get data from
    :param queue_to_post_process: queue to put cropped and resized data
    :param n_processes:
    :return:
    """
    with Pool(n_processes) as pool:
        filenames, predictions = queue_from_gpu.get()

        probs = predictions['probs']
        original_shapes = predictions['original_shape']
        resized_shapes = predictions['resized_shape']

        pool.starmap(_crop_and_resize, [(filenames[i],
                                         probs[i, :, :, :],
                                         resized_shapes[i],
                                         original_shapes[i],
                                         queue_to_post_process) for i in range(len(filenames))])

    queue_from_gpu.task_done()


def resizing_process_batch(queue_from_gpu: JoinableQueue,
                           queue_to_post_process: JoinableQueue,
                           stopping_event: Event,
                           n_processes: int=None):
    """
    Resizing Process when batch prediction is used

    :param queue_from_gpu:
    :param queue_to_post_process:
    :param stopping_event: Event that indicated end of GPU prediction (no more elements are
        enqueued to `queue_from_gpu`)
    :param n_processes:
    :return:
    """

    with ThreadPool(int(n_processes/2)) as threadpool:
        time_slept = 0
        while True:
            if not stopping_event.is_set() or not queue_from_gpu.empty():
                time_slept = 0
                threadpool.apply(_wk_crop_and_resize_from_queue, (queue_from_gpu, queue_to_post_process, n_processes,))
            else:
                time.sleep(1)
                time_slept += 1

                if time_slept >= _MAX_SLEPT_TIME:
                    break

        queue_from_gpu.join()


# ---------------------------------------------------


def post_process_after_resizing(queue: JoinableQueue,
                                wk_extraction_fn,
                                output_page_dir: str,
                                model_dir: str,
                                stopping_event: Event,
                                n_processes: int = 8):
    """

    :param queue:
    :param output_page_dir:
    :param model_dir:
    :param stopping_event:
    :param n_processes:
    :return:
    """

    time_slept = 0
    with Pool(n_processes) as pool:
        while True:
            if not stopping_event.is_set() or not queue.empty():
                time_slept = 0
                pool.apply_async(wk_extraction_fn, (queue, output_page_dir, model_dir,))
            else:
                time.sleep(1)
                time_slept += 1

                if time_slept >= _MAX_SLEPT_TIME:
                    break
        queue.join()


def post_process_without_resizing_batch(queue: JoinableQueue,
                                        wk_extraction_fn,
                                        output_page_dir: str,
                                        model_dir: str,
                                        stopping_event: Event,
                                        n_processes: int=8):
    """

    :param queue:
    :param output_page_dir:
    :param model_dir:
    :param stopping_event:
    :param n_processes:
    :return:
    """

    time_slept = 0
    with ThreadPool(int(n_processes/2)) as threadpool:
        while True:
            if not stopping_event.is_set() or not queue.empty():
                time_slept = 0
                threadpool.apply_async(wk_extraction_fn, (queue,
                                                          output_page_dir,
                                                          model_dir,
                                                          n_processes,))
            else:
                time.sleep(1)
                time_slept += 1

                if time_slept >= _MAX_SLEPT_TIME:
                    break
        queue.join()
