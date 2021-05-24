import json
import os

import nibabel as nib
import numpy as np
import tensorflow as tf
from keras.models import load_model

from pipeline.constants import DirectoryStructure as ds, FileNames as fn
from pipeline.logger import log
from pipeline.plot import landmarks_as_spheres
from pipeline.register import atlas_landmarks
from pipeline.util import get_project_root, normalize

ROOT = get_project_root()
ATLAS_PATH = os.path.join(str(ROOT), 'atlas')
MODEL_PATH = os.path.join(str(ROOT), 'models')


def landmarks_as_dict(file_name: str) -> dict:

    if file_name is None or not os.path.exists(file_name):
        return {}
    landmark_data = nib.load(file_name).get_data()
    unique_values = np.unique(landmark_data)[1:]
    pts = np.argwhere(landmark_data)
    output = dict()
    for label in unique_values:
        output[str(label)] = pts[landmark_data[pts[:, 0], pts[:, 1], pts[:, 2]] == label].mean(axis=0)

    return output


def predict_bone_joint_model(n_images: int = 35, radius: int = 32) -> None:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf.keras.backend.set_session(tf.Session(config=config))

    fat = nib.load(ds.nifti.path(fn.fat.value))
    x, y, z = fat.shape
    np_landmarks = np.zeros((x, y, z), dtype='uint8')

    atlas_landmarks(ATLAS_PATH, n_images)
    landmarks_initial = landmarks_as_dict(ds.analysis.path(fn.bone_joints_initial_mask.value))
    cube_of_zeros = np.zeros((2 * radius, 2 * radius, 2 * radius), dtype='uint16')
    landmarks_dict = dict()

    model_shoulder = load_model(os.path.join(MODEL_PATH, 'shoulder_model.h5'))
    model_hip = load_model(os.path.join(MODEL_PATH, 'hip_model.h5'))
    model_knee = load_model(os.path.join(MODEL_PATH, 'knee_model.h5'))

    for label in ['1', '2', '11', '12', '21', '22']:
        if label == '1':
            bone_joint = 'shoulder'
            hand = 'right'
            model = model_shoulder
        elif label == '2':
            bone_joint = 'shoulder'
            hand = 'left'
            model = model_shoulder
        elif label == '11':
            bone_joint = 'hip'
            hand = 'right'
            model = model_hip
        elif label == '12':
            bone_joint = 'hip'
            hand = 'left'
            model = model_hip
        elif label == '21':
            bone_joint = 'knee'
            hand = 'right'
            model = model_knee
        elif label == '22':
            bone_joint = 'knee'
            hand = 'left'
            model = model_knee
        else:
            log.error(f"The label '{label}' is not valid")
            return
        landmarks_count = np.sum(nib.load(ds.analysis.path(fn.bone_joints_initial_mask.value)).get_data() == int(label))
        if landmarks_count < n_images // 2:
            log.warning(f'Too few landmarks are present from the registration step, skipping {hand} {bone_joint}')
            continue
        else:
            log.debug(f'Predicting bone joint for {hand} {bone_joint}')
        lx, ly, lz = landmarks_initial[label].astype('uint16')
        crange = {
            'x': slice(0 - min(0, lx - radius), 2 * radius + min(0, x - (lx + radius))),
            'y': slice(0 - min(0, ly - radius), 2 * radius + min(0, y - (ly + radius))),
            'z': slice(0 - min(0, lz - radius), 2 * radius + min(0, z - (lz + radius)))
        }
        drange = {
            'x': slice(max(0, lx - radius), min(x, lx + radius)),
            'y': slice(max(0, ly - radius), min(y, ly + radius)),
            'z': slice(max(0, lz - radius), min(z, lz + radius))
        }
        fat_data_crop = cube_of_zeros.copy()
        fat_data_crop[crange['x'], crange['y'], crange['z']] = fat.get_data()[drange['x'], drange['y'], drange['z']]
        fat_data_crop = normalize(fat_data_crop)
        coordinates = model.predict(fat_data_crop[np.newaxis, :, :, :, np.newaxis]).astype('uint16').flatten()
        if (coordinates >= 0).all() and (coordinates <= 2 * radius).all():
            label_crop = cube_of_zeros.copy()
            label_crop[coordinates[0], coordinates[1], coordinates[2]] = int(label)
            np_landmarks[drange['x'], drange['y'], drange['z']] = label_crop[crange['x'], crange['y'], crange['z']]
            if not np.argwhere(np_landmarks == int(label)).flatten().tolist():
                log.warning('Estimated bone joint outside volume!')
            else:
                landmarks_dict[label] = np.argwhere(np_landmarks == int(label)).flatten().tolist()
        else:
            log.warning(f'Estimated bone joint {coordinates} outside ROI!')

        del model

    # Save NIfTI file
    if not ds.landmarks.exists():
        os.makedirs(ds.landmarks.value)
    nib.Nifti1Image(np_landmarks, fat.affine, fat.header).to_filename(ds.landmarks.path(fn.bone_joints_mask.value))
    spheres = landmarks_as_spheres(np_landmarks, radius=16, spacing=fat.header['pixdim'][1:4])
    nib.Nifti1Image(spheres, fat.affine, fat.header).to_filename(ds.landmarks.path(fn.bone_joints_spheres.value))

    # Save dictionary to JSON file
    with open(ds.landmarks.path(fn.bone_joints_json.value), 'w') as landmarks_json_file:
        json.dump(landmarks_dict, landmarks_json_file, sort_keys=True, indent=4)

    return
