import glob
import json
import os
import re
import shutil
import tempfile
import zipfile
from typing import Union, Dict

import numpy as np
from dcmstack import parse_and_stack, InvalidStackError
from pydicom import dcmread

from pipeline.constants import DirectoryStructure as ds, VALID_BIOBANK_PROJECTS
from pipeline.logger import log
from pipeline.util import channel_from_image_type

ABDOMINAL_PROTOCOL = ['20201', '20202', '20203', '20206', '20254', '20259', '20260']
DICOM_HEADER_FIELDS = ('SeriesNumber', 'SeriesDescription', 'ImageType', 'PatientName', 'PatientID')
DICOM_ID = 'UNKNOWN'


def create_dicom_id(patient_name: str, patient_id: str) -> str:
    if patient_name == patient_id:
        dicom_id = re.sub('\^Bio', '', str(patient_name).replace(' ', ''))
    else:
        dicom_id = '{}{}'.format(re.sub('\^Bio', '', str(patient_name).replace(' ', '')),
                                 str(patient_id).replace(' ', ''))

    return dicom_id


def create_biobank_id(path_name: str) -> str:
    # Assuming the Biobank ID is 7 adjacent digits somewhere in the pathname
    p = re.compile('([0-9][0-9][0-9][0-9][0-9][0-9][0-9])')
    result = p.search(path_name)

    return result.group(1) if result else ''


def is_dicom(file_name):
    with open(file_name, 'rb') as file:
        return file.read(133)[128:132] == b'DICM'


def is_biobank_project_valid(project_id: Union[str, None]) -> bool:
    return project_id in VALID_BIOBANK_PROJECTS


def unzip_dicom_data(dicom_directory: str) -> (str, str, bool):
    if dicom_directory.endswith('zip'):
        biobank_id = os.path.basename(dicom_directory).split('_')[0]
        temp_directory = tempfile.mkdtemp()
        for zip_file in glob.glob(os.path.join(os.path.dirname(dicom_directory), '{}*.zip'.format(biobank_id))):
            extract_directory = os.path.join(temp_directory, os.path.basename(zip_file).replace('.zip', ''))
            log.info('Unzipping {} into {}'.format(os.path.basename(zip_file), extract_directory))
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_directory)
        return temp_directory, biobank_id, True

    return dicom_directory, '', False


def organize_dicom_files(input_directory: str, output_directory: str, biobank_project: Union[str, None]) -> (str, str):
    input_directory, bid, zip_data = unzip_dicom_data(input_directory)
    biobank_id = ''
    if biobank_project:
        if not is_biobank_project_valid(biobank_project):
            log.error('Project ID {} is not valid'.format(biobank_project))
            raise RuntimeError
        biobank_id = bid if zip_data else create_biobank_id(input_directory)
        log.info('UK Biobank project: {}'.format(biobank_project))
        log.info('UK Biobank ID: {}'.format(biobank_id))
    dicom_id = ''
    for root, dirs, files in os.walk(input_directory):
        for file_name in files:
            file_with_path = os.path.join(root, file_name)
            if is_dicom(file_with_path):
                dcm = dcmread(file_with_path, stop_before_pixels=True, force=False)
                if dicom_id == '':
                    dicom_id = create_dicom_id(dcm.PatientName, dcm.PatientID)
                new_dir_path = os.path.join(os.path.expanduser(output_directory),
                                            dicom_id, ds.tmp_dicom_series.value,
                                            '{:04d}_{}'.format(int(dcm.SeriesNumber), dcm.SeriesDescription))
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                shutil.copyfile(file_with_path, os.path.join(new_dir_path, file_name))

    log.info('DICOM files organized for {}'.format(dicom_id))
    global DICOM_ID
    DICOM_ID = dicom_id

    if zip_data:
        log.info('Removing temporary directory holding DICOM data')
        shutil.rmtree(input_directory, ignore_errors=True)

    return dicom_id, biobank_id


def bad_nifti_series_handler(dicom_id: str) -> None:
    # Add appropriate DICOM IDs as they are identified
    if dicom_id == '89TZ72J7ANDA5QUN':
        if not ds.tmp_nifti_series_donotuse.exists():
            os.makedirs(ds.tmp_nifti_series_donotuse.value)
        dixon = ['0021_mag_Dixon_noBH_in.nii.gz',
                 '0022_mag_Dixon_noBH_opp.nii.gz',
                 '0023_mag_Dixon_noBH_F.nii.gz',
                 '0024_mag_Dixon_noBH_W.nii.gz']
        for i in range(len(dixon)):
            shutil.move(ds.tmp_nifti_series.path(dixon[i]), ds.tmp_nifti_series_donotuse.path(dixon[i]))
        return

    files = list()
    n_dixon_series = 0
    if ds.tmp_nifti_series.exists():
        for _, _, files in os.walk(ds.tmp_nifti_series.value):
            for file in files:
                n_dixon_series += 1 if 'dixon' in file.lower() else 0
    n_to_remove = n_dixon_series - 24
    if n_to_remove > 0:
        log.info('Too many DICOM series, attempting to remove excess and proceed')
        if not ds.tmp_nifti_series_donotuse.exists():
            os.makedirs(ds.tmp_nifti_series_donotuse.value)
        dixon_files = [file_name for file_name in files if 'dixon' in file_name.lower()]
        dixon_files.sort()
        for index in range(n_to_remove):
            shutil.move(ds.tmp_nifti_series.path(dixon_files[index]),
                        ds.tmp_nifti_series_donotuse.path(dixon_files[index]))


def process_dicom_series(dcm_series: Dict, input_directory: str, nifti_directory: str, biobank_project: str,
                         biobank_id: str, dicom_id: str = '') -> str:
    for key, volume in sorted(dcm_series.items()):
        series_name = key[1]
        try:
            series_number = int(key[0])
        except TypeError:
            log.error("Cannot find series id for series '{}'".format(key))
            continue

        try:
            shape = volume.get_shape()
        except InvalidStackError as error_message:
            log.error('InvalidStackError: {}'.format(error_message))
            log.debug('Difference between slices\n{}'.format(np.diff(list(volume._slice_pos_vals))))
            continue

        if np.greater(shape, 1).sum() < 3:
            log.warning('Skipping 2D series {:04d}_{} [{}]'.format(series_number, series_name, shape))
            continue
        nifti_volume = volume.to_nifti_wrapper(voxel_order='LAS')
        if dicom_id == '':
            dicom_id = create_dicom_id(key[3], key[4])
            log.info('DICOM subject name: {}'.format(dicom_id))
            output_directory = os.path.join(nifti_directory, dicom_id, ds.summary.value)
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            if os.path.exists(os.path.join(output_directory, 'metadata.json')):
                with open(os.path.join(output_directory, 'metadata.json'), 'r') as jsonfile:
                    metadata = json.load(jsonfile)
            else:
                metadata = dict()
            metadata['input_directory'] = input_directory
            metadata['biobank_project'] = biobank_project
            with open(os.path.join(output_directory, 'metadata.json'), 'w') as jsonfile:
                json.dump(metadata, jsonfile, sort_keys=True, indent=4, ensure_ascii=False)

        spacing = nifti_volume.nii_img.header.get_zooms()
        residual = nifti_volume.nii_img.affine[:3, :3] - np.diag((-spacing[0], spacing[1], spacing[2]))
        if not np.allclose(0, residual, atol=1e-3):
            log.warning('Affine residual not zero')

        log.debug('Saving {:04d}_{}'.format(series_number, series_name))
        output_directory = os.path.join(nifti_directory, dicom_id, ds.tmp_nifti_series.value)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        file_name = os.path.join(output_directory, '{:04d}_{}_{}.nii.gz'.format(series_number,
                                                                                channel_from_image_type(key[2]),
                                                                                re.sub(r'\W+', '', series_name)))
        # Sneak some JSON info into the NIfTI-1 header
        nifti_volume.nii_img.header.structarr['descrip'] = \
            '{{"dicom":"{}", "biobank":"{}", "project":"{}"}}'.format(dicom_id, biobank_id, biobank_project)
        nifti_volume.to_filename(file_name)

    return dicom_id


def dicom_to_nifti(dicom_directory: str, nifti_directory: str, biobank_project: Union[str, None],
                   biobank_id: str = '') -> (str, str):
    input_directory = dicom_directory
    dicom_directory, bid, zip_data = unzip_dicom_data(dicom_directory)

    if biobank_project:
        if not is_biobank_project_valid(biobank_project):
            log.error('Biobank project ID {} is not valid'.format(biobank_project))
            raise RuntimeError
        biobank_id = bid if zip_data else create_biobank_id(dicom_directory)
        log.info('UK Biobank project: {}'.format(biobank_project))
        log.info('UK Biobank ID: {}'.format(biobank_id))

    dicom_id = ''
    list_of_paths = list()
    unwanted_files = list()
    for root, dirs, files in os.walk(dicom_directory):
        for file_name in files:
            file_with_root = os.path.join(root, file_name)
            if is_dicom(file_with_root) and 'report' not in file_with_root.lower() and \
                    'localizer' not in file_with_root.lower() and 'scout' not in file_with_root.lower():
                list_of_paths.append(file_with_root)
            else:
                unwanted_files.append(file_with_root)
    log.info('Read {} DICOM files ({} files ignored)'.format(len(list_of_paths), len(unwanted_files)))

    if biobank_project:
        for field in ABDOMINAL_PROTOCOL:
            dcm_series = None
            list_of_paths_for_field = [x for x in list_of_paths if '_{}_'.format(field) in x]
            if len(list_of_paths_for_field) > 0:
                log.debug('dcmstack.parse_and_stack for field {}'.format(field))
                try:
                    dcm_series = parse_and_stack(list_of_paths_for_field, group_by=DICOM_HEADER_FIELDS)
                except ValueError as error_message:
                    log.error('ValueError: {}'.format(error_message))
                except:
                    log.error('Some other type of error')
            if not dcm_series:
                continue

            dicom_id = process_dicom_series(dcm_series, input_directory, nifti_directory, biobank_project,
                                            biobank_id, dicom_id)
    else:
        dcm_series = parse_and_stack(list_of_paths, group_by=DICOM_HEADER_FIELDS)

        if len(dcm_series.keys()) == 0:
            log.error('No DICOM series found')
            return dicom_id

        dicom_id = process_dicom_series(dcm_series, input_directory, nifti_directory, biobank_project, biobank_id,
                                        dicom_id)

    if biobank_project is not None:
        os.chdir(os.path.join(nifti_directory, dicom_id))
        bad_nifti_series_handler(dicom_id)

    if zip_data:
        log.info('Removing temporary directory with DICOM data')
        shutil.rmtree(dicom_directory, ignore_errors=True)

    return dicom_id, biobank_id
