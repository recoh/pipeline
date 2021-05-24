import json

import SimpleITK as sitk
import cv2 as cv
import numpy as np
from scipy import ndimage
from skimage import measure

from pipeline.constants import DirectoryStructure as ds, FileNames as fn
from pipeline.logger import log


def auto_canny(image: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    v = np.median(image)

    return cv.Canny(image, int(max(0, (1.0 - sigma) * v)), int(min(255, (1.0 + sigma) * v)))


def anomaly_detection() -> None:
    crop_val = 20
    volume = sitk.ReadImage(ds.analysis.path(fn.body_mask.value))
    x, y, z = volume.GetSize()
    volume = sitk.GetArrayFromImage(volume)

    c = np.zeros([1, y])
    for d in range(y):
        c[0, d] = volume[:, d, :].sum()
    coronal = volume[::-1, c.argmax(), :]

    s = np.zeros([1, x])
    for d in range(x):
        s[0, d] = volume[:, :, d].sum()
    sagittal = volume[::-1, :, s.argmax()]

    a = np.nonzero(coronal)
    firstx = a[0][0]
    lastx = a[0][len(a[0]) - 1]

    coronal_crop = coronal[firstx + crop_val:lastx - crop_val, :]
    edges_coronal = auto_canny(coronal_crop)
    dummy = np.float32([[1, 0, 4], [0, 1, 0]])
    rows, cols = edges_coronal.shape[:2]
    shifted = cv.warpAffine(edges_coronal, dummy, (cols, rows))
    shifted = shifted > 0
    shifted = shifted * 1
    edge_dummy = edges_coronal > 1
    edge_dummy = edge_dummy * 1
    remove_pix_c = shifted + edge_dummy > 1
    remove_pix_c = remove_pix_c * 1
    eroded_c = ndimage.binary_erosion(remove_pix_c, structure=np.ones((1, 2)))
    c_erodedsum = eroded_c.sum()

    sagittal_crop = sagittal[firstx + crop_val:lastx - crop_val, :]
    edges_sagittal = auto_canny(sagittal_crop)
    dummy = np.float32([[1, 0, 4], [0, 1, 0]])
    rows, cols = edges_sagittal.shape[:2]
    shifted = cv.warpAffine(edges_sagittal, dummy, (cols, rows))
    shifted = shifted > 0
    shifted = shifted * 1
    edge_dummy = edges_sagittal > 1
    edge_dummy = edge_dummy * 1
    remove_pix_s = shifted + edge_dummy > 1
    remove_pix_s = remove_pix_s * 1
    eroded_s = ndimage.binary_erosion(remove_pix_s, structure=np.ones((1, 2)))
    s_erodedsum = eroded_s.sum()

    labels_c = measure.label(eroded_c)
    props_c = measure.regionprops(labels_c)

    labels_s = measure.label(eroded_s)
    props_s = measure.regionprops(labels_s)

    if len(props_c) > 10 and len(props_s) > 10 and c_erodedsum > 20 or s_erodedsum > 20:
        flag_mask_hole = True
    else:
        flag_mask_hole = False

    if z < 370:
        flag_mask_size = True
    else:
        flag_mask_size = False

    if (c_erodedsum > 10 and s_erodedsum > 10) or (c_erodedsum > 25 or s_erodedsum > 25) or flag_mask_hole:
        log.warning('There is a data issue')
        decision = 'Issue detected, please verify'
    elif flag_mask_size:
        msg = 'series_missing'
        log.warning('There is a data issue ({})'.format(msg))
        decision = 'Issue detected, please verify'
    else:
        decision = ''
    if ds.summary.exists('metadata.json'):
        with open(ds.summary.path('metadata.json'), 'r') as jsonfile:
            metadata = json.load(jsonfile)
    metadata['anomaly_detection'] = decision

    with open(ds.summary.path('metadata.json'), 'w') as jsonfile:
        json.dump(metadata, jsonfile, sort_keys=True, indent=4, ensure_ascii=False)
