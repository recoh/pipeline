import copy
import json
import os
from typing import List, Union, Dict

import numpy as np
from keras.engine.saving import load_model

from pipeline.constants import DataChannel as dc, DirectoryStructure as ds
from pipeline.logger import log
from pipeline.nifti_series import create_midpoint_by_series, replace_nii_img_dataobj, add_series_to_dict
from pipeline.util import get_project_root

ROOT = get_project_root()
MODELS_DIRECTORY = os.path.join(str(ROOT), 'models')
SERIES_DIM = {
    '1': [224, 168, 64],
    '2': [224, 174, 44],
    '3': [224, 174, 44],
    '4': [224, 174, 44],
    '5': [224, 162, 72],
    '6': [224, 156, 64]
}


def swap_detection(series_data: Dict) -> Dict:
    model_dict = {
        'c1': load_model(os.path.join(MODELS_DIRECTORY, 'classify_series_1.h5')),
        'c2': load_model(os.path.join(MODELS_DIRECTORY, 'classify_series_2.h5')),
        'c3': load_model(os.path.join(MODELS_DIRECTORY, 'classify_series_3.h5')),
        'c4': load_model(os.path.join(MODELS_DIRECTORY, 'classify_series_4.h5')),
        'c5_1': load_model(os.path.join(MODELS_DIRECTORY, 'classify_series_5_L.h5')),
        'c5_2': load_model(os.path.join(MODELS_DIRECTORY, 'classify_series_5_R.h5')),
        'c6_1': load_model(os.path.join(MODELS_DIRECTORY, 'classify_series_6_L.h5')),
        'c6_2': load_model(os.path.join(MODELS_DIRECTORY, 'classify_series_6_R.h5'))
    }

    swap_detected = dict()
    metadata = dict()
    if ds.summary.exists('metadata.json'):
        with open(ds.summary.path('metadata.json'), 'r') as jsonfile:
            metadata = json.load(jsonfile)
    metadata['swap_correction'] = None

    log.info('Swap detection using neural networks')
    fat_data = series_data[dc.fat]
    water_data = series_data[dc.water]
    sorted_keys = sorted(fat_data.keys(), reverse=True)

    for i in range(min(len(sorted_keys), 5)):
        water_tf, fat_tf = preprocess_swaps(i + 1, SERIES_DIM[str(i + 1)], sorted_keys, water_data, fat_data)
        fat_water_array = np.zeros([[water_tf.shape[0]][0], [water_tf.shape[1]][0], [water_tf.shape[2]][0], 2])

        # fat_water_array has fat in channel 0 and water in channel 1.
        # accordingly, fat has label 0 and water has label 1.
        fat_water_array[:, :, :, 0] = fat_tf[:, :, :, 0]
        fat_water_array[:, :, :, 1] = water_tf[:, :, :, 0]
        swap_detected[sorted_keys[i]] = predict_swaps(fat_water_array, model_dict, i + 1)
    # Series 6
    if len(sorted_keys) >= 6:
        water_tf, fat_tf = preprocess_swaps(6, SERIES_DIM['6'], sorted_keys, water_data, fat_data)

        fat_water_array = np.zeros([[water_tf.shape[0]][0], [water_tf.shape[1]][0], [water_tf.shape[2]][0], 2])
        fat_water_array[:, :, :, 0] = fat_tf[:, :, :, 0]
        fat_water_array[:, :, :, 1] = water_tf[:, :, :, 0]

        swap_detected[sorted_keys[5]] = predict_swaps(fat_water_array, model_dict, 6)
    else:
        log.warning('Series 6 is not present, skipping')

    metadata['swap_detected'] = swap_detected
    if not ds.summary.exists():
        os.mkdir(ds.summary.value)
    with open(ds.summary.path('metadata.json'), 'w') as jsonfile:
        json.dump(metadata, jsonfile, sort_keys=True, indent=4, ensure_ascii=False)

    return swap_detected


def normalize(img: np.ndarray) -> np.ndarray:
    img = img.astype('float32') / np.percentile(img, 99)
    img[img > 1.0] = 0.975
    return img


def preprocess_swaps(sn: int, s_dim: List, sorted_keys: List, water_data: dict, fat_data: dict) -> \
        (np.ndarray, np.ndarray):
    x, y, z = s_dim
    fat_vol = fat_data[sorted_keys[sn - 1]].nii_img
    water_vol = water_data[sorted_keys[sn - 1]].nii_img
    if list(fat_vol.shape) != s_dim:
        log.error('Dimensions do not match training data')
        return
    fat_vol = fat_vol.get_data()
    water_vol = water_vol.get_data()
    fatd = normalize(fat_vol)
    s = np.zeros([1, y])
    for d in range(y):
        dummy = fatd[:, d, :] > 0.1
        s[0, d] = dummy.sum()
    out_tpl = np.nonzero(s)
    mean_index = (out_tpl[1][0] + out_tpl[1][-1]) / 2
    index = int((mean_index + np.argmax(s)) // 2)  # extract 2D slice to run test on
    water_slice = normalize(water_vol[:, index, :])  # reshape to tf shape
    water_slice = water_slice.transpose((1, 0))
    water_tf = water_slice[np.newaxis, ..., np.newaxis]
    fat_slice = normalize(fat_vol[:, index, :])
    fat_slice = fat_slice.transpose((1, 0))
    fat_tf = fat_slice[np.newaxis, ..., np.newaxis]

    # outputs: central water and fat slices in tensorflow shape
    return water_tf, fat_tf


def predict_swaps(fat_water_array: np.ndarray, models: dict, series_num: int) -> Union[bool, dict]:
    swap_detected = False  # no swap detected unless otherwise stated
    if series_num == 1 or series_num == 2 or series_num == 3 or series_num == 4:
        test_result = models['c{}'.format(series_num)].predict(fat_water_array)
        fat_test = float(test_result[0][0])
        water_test = float(test_result[0][1])

        # the model assesses both images and assigns one unique label to each image
        # therefore there is a swap if and only if both labels are found to be wrong.
        # we expect fat = 0 and water = 1
        if round(water_test) == 0 and round(fat_test) == 1:
            log.warning('Swap identified in series {}'.format(series_num))
            swap_detected = True

        return swap_detected

    _, x, y, _ = fat_water_array.shape
    # series 5 and 6, applied on the left and right halves of the image (on each leg)
    if series_num == 5 or series_num == 6:

        test_result_1 = (models['c{}_1'.format(series_num)].predict(fat_water_array[:, :, 0:(y // 2), :]))
        fat_test_1 = float(test_result_1[0][0])
        water_test_1 = float(test_result_1[0][1])

        test_result_2 = (models['c{}_2'.format(series_num)].predict(fat_water_array[:, :, (y // 2):y, :]))
        fat_test_2 = float(test_result_2[0][0])
        water_test_2 = float(test_result_2[0][1])

        # no swap detected unless otherwise stated
        swap_detected = {'left': False, 'right': False}

        # the model assesses both images and assigns one unique label to each image
        # therefore there is a swap if and only if both labels are found to be wrong.
        # we expect fat = 0 and water = 1
        if round(water_test_1) == 0 and round(fat_test_1) == 1:
            log.warning('Swap identified in series {} (right)'.format(series_num))
            swap_detected['right'] = True
        if round(water_test_2) == 0 and round(fat_test_2) == 1:
            log.warning('Swap identified in series {} (left)'.format(series_num))
            swap_detected['left'] = True

        return swap_detected


def any_swaps(swaps_detected: Dict) -> bool:
    def nested_dict_values(d):
        for v in d.values():
            if isinstance(v, dict):
                yield from nested_dict_values(v)
            else:
                yield v

    return any(list(nested_dict_values(swaps_detected)))


def fix_fat_water_swaps(series_data: Dict, swaps: Dict) -> Dict:
    create_midpoint_by_series(series_data)
    fat_series = series_data[dc.fat]
    water_series = series_data[dc.water]

    metadata = dict()
    if ds.summary.exists('metadata.json'):
        with open(ds.summary.path('metadata.json'), 'r') as jsonfile:
            metadata = json.load(jsonfile)
    metadata['swap_correction'] = {key: None for key in fat_series.keys()}

    series_data_swapped = copy.deepcopy(series_data)
    for location in fat_series.keys():
        fat_data = np.asanyarray(fat_series[location].nii_img.dataobj)
        fat_shape = fat_data.shape
        water_data = np.asanyarray(water_series[location].nii_img.dataobj)
        if isinstance(swaps[location], dict) and not (swaps[location]['left'] and swaps[location]['right']):
            np_midpoint_labels = np.asanyarray(series_data[dc.midpoint][location].nii_img.dataobj)
            for hand, is_swapped in swaps[location].items():
                if is_swapped:
                    if hand == 'left':
                        log.info('Swapping the left half of the series at location {}'.format(location))
                        new_fat_data = np.zeros(fat_shape)
                        new_fat_data[np_midpoint_labels == 0] = fat_data[np_midpoint_labels == 0]
                        new_fat_data[np_midpoint_labels == 1] = water_data[np_midpoint_labels == 1]
                        new_fat_data[np_midpoint_labels == 2] = fat_data[np_midpoint_labels == 2]

                        new_water_data = np.zeros(fat_shape)
                        new_water_data[np_midpoint_labels == 0] = water_data[np_midpoint_labels == 0]
                        new_water_data[np_midpoint_labels == 1] = fat_data[np_midpoint_labels == 1]
                        new_water_data[np_midpoint_labels == 2] = water_data[np_midpoint_labels == 2]
                    else:
                        log.info('Swapping the right half of the series at location {}'.format(location))
                        new_fat_data = np.zeros(fat_shape)
                        new_fat_data[np_midpoint_labels == 0] = fat_data[np_midpoint_labels == 0]
                        new_fat_data[np_midpoint_labels == 1] = fat_data[np_midpoint_labels == 1]
                        new_fat_data[np_midpoint_labels == 2] = water_data[np_midpoint_labels == 2]

                        new_water_data = np.zeros(fat_shape)
                        new_water_data[np_midpoint_labels == 0] = water_data[np_midpoint_labels == 0]
                        new_water_data[np_midpoint_labels == 1] = water_data[np_midpoint_labels == 1]
                        new_water_data[np_midpoint_labels == 2] = fat_data[np_midpoint_labels == 2]

                    series_data_swapped = \
                        add_series_to_dict(replace_nii_img_dataobj(water_series[location], new_water_data),
                                           series_data_swapped, dc.water, location)
                    series_data_swapped = \
                        add_series_to_dict(replace_nii_img_dataobj(fat_series[location], new_fat_data),
                                           series_data_swapped, dc.fat, location)
                    metadata['swap_correction'][location] = True
        elif swaps[location]:
            log.info('Swapping the whole series at location {}'.format(location))
            series_data_swapped = add_series_to_dict(water_series[location], series_data_swapped, dc.fat, location)
            series_data_swapped = add_series_to_dict(fat_series[location], series_data_swapped, dc.water, location)
            metadata['swap_correction'][location] = True

    log.info('Swap correction = {}'.format(metadata['swap_correction']))

    if not ds.summary.exists():
        os.mkdir(ds.summary.value)
    with open(ds.summary.path('metadata.json'), 'w') as jsonfile:
        json.dump(metadata, jsonfile, sort_keys=True, indent=4, ensure_ascii=False)

    return series_data_swapped
