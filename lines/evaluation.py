#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import io
import os
import subprocess
import pandas as pd
from tqdm import tqdm
from typing import List

CBAD_JAR = './cBAD/TranskribusBaseLineEvaluationScheme_v0.1.3/' \
           'TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar'
# PP_PARAMS = post_process_params = {'sigma': 1.5, 'low_threshold': 0.2, 'high_threshold': 0.4}


def eval_fn(label_filenames: List[str],
            prediction_xml_dir: str,
            groundtruth_dir: str,
            output_dir: str=None,
            jar_tool_path: str=CBAD_JAR) -> dict:
    """

    :param label_filenames: Input directory containing probability maps (.npy)
    :param prediction_xml_dir:
    :param groundtruth_dir: directory containg XML groundtruths
    :param output_dir: output directory for results
    :param jar_tool_path: path to cBAD evaluation tool (.jar file)
    :return:
    """

    if output_dir is None:
        output_dir = os.path.join(prediction_xml_dir, 'eval_output')
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    else:
        "WARNING - Output dir {} already exists".format(output_dir)

    assert(os.path.isdir(groundtruth_dir)), "Groundtruth dir does not exist at {}".format(groundtruth_dir)

    # Take basename and remove extension from filename
    basenames_labels = map(lambda x: os.path.basename(x).split('.')[0], label_filenames)

    xml_filenames_tuples = list()
    for elem in tqdm(basenames_labels):
        prediction_xml_filename = os.path.join(prediction_xml_dir, elem + '.xml')
        gt_xml_filename = os.path.join(groundtruth_dir, elem + '.xml')
        assert os.path.isfile(prediction_xml_filename), "{} does not exist".format(prediction_xml_filename)
        assert os.path.isfile(gt_xml_filename), "{} does not exist".format(gt_xml_filename)

        xml_filenames_tuples.append((gt_xml_filename, prediction_xml_filename))

    gt_pages_list_filename = os.path.join(output_dir, 'gt_pages_simple.lst')
    generated_pages_list_filename = os.path.join(output_dir, 'generated_pages_simple.lst')
    with open(gt_pages_list_filename, 'w') as f:
        f.writelines('\n'.join([s[0] for s in xml_filenames_tuples]))
    with open(generated_pages_list_filename, 'w') as f:
        f.writelines('\n'.join([s[1] for s in xml_filenames_tuples]))

    # Evaluation using JAVA Tool
    cmd = 'java -jar {} {} {}'.format(jar_tool_path, gt_pages_list_filename, generated_pages_list_filename)
    result = subprocess.check_output(cmd, shell=True).decode()
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as f:
        f.write(result)
    parse_score_txt(result, os.path.join(output_dir, 'scores.csv'))

    # Parse results from output of tool
    lines = result.splitlines()
    avg_precision = float(next(filter(lambda l: 'Avg (over pages) P value:' in l, lines)).split()[-1])
    avg_recall = float(next(filter(lambda l: 'Avg (over pages) R value:' in l, lines)).split()[-1])
    f_measure = float(next(filter(lambda l: 'Resulting F_1 value:' in l, lines)).split()[-1])

    print('P {}, R {}, F {}'.format(avg_precision, avg_recall, f_measure))

    return {
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'f_measure': f_measure
    }


def parse_score_txt(score_txt, output_csv):
    lines = score_txt.splitlines()
    header_ind = next((i for i, l in enumerate(lines)
                       if l == '#P value, #R value, #F_1 value, #TruthFileName, #HypoFileName'))
    final_line = next((i for i, l in enumerate(lines) if i > header_ind and l == ''))
    csv_data = '\n'.join(lines[header_ind:final_line])
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.rename(columns={k: k.strip() for k in df.columns})
    df['#HypoFileName'] = [os.path.basename(f).split('.')[0] for f in df['#HypoFileName']]
    del df['#TruthFileName']
    df = df.rename(columns={'#P value': 'P', '#R value': 'R', '#F_1 value': 'F_1', '#HypoFileName': 'basename'})
    df = df.reindex(columns=['basename', 'F_1', 'P', 'R'])
    df = df.sort_values('F_1', ascending=True)
    df.to_csv(output_csv, index=False)
