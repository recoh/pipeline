import json
import os
import time

import nibabel as nib
from pydicom import dcmread

from pipeline.constants import DirectoryStructure as ds
from pipeline.dicom import create_dicom_id


def subject_metadata() -> None:
    metadata = dict()
    if ds.summary.exists('metadata.json'):
        with open(ds.summary.path('metadata.json'), 'r') as jsonfile:
            metadata = json.load(jsonfile)

    # DICOM information
    n_dicom_files = 0
    for root, dirs, files in os.walk(ds.tmp_dicom_series.value):
        n_dicom_files += len(files)
        for file_name in files:
            dicom_file = os.path.join(root, file_name)
    metadata['n_dicom_files'] = n_dicom_files
    metadata['n_dicom_dirs'] = sum([len(dirs) for _, dirs, _ in os.walk(ds.tmp_dicom_series.value)])
    dcm = dcmread(dicom_file)
    metadata['dicom_id'] = create_dicom_id(dcm.PatientName, dcm.PatientID)
    metadata['patient_size'] = dcm.PatientSize
    metadata['patient_weight'] = dcm.PatientWeight
    metadata['study_timestamp'] = \
        time.strftime('%d/%m/%Y %X', time.strptime('{} {}'.format(dcm.StudyDate, dcm.StudyTime), '%Y%m%d %H%M%S.%f'))
    if 'Stockport' in dcm.InstitutionAddress:
        metadata['scanning_site'] = 'Stockport'
    elif 'Newcastle' in dcm.InstitutionAddress:
        metadata['scanning_site'] = 'Newcastle'
    elif 'Reading' in dcm.InstitutionAddress:
        metadata['scanning_site'] = 'Reading'
    elif 'Bristol' in dcm.InstitutionAddress:
        metadata['scanning_site'] = 'Bristol'
    else:
        metadata['scanning_site'] = 'Unknown'

    # NIfTI information
    n_dixon_series = 0
    if ds.tmp_nifti_series.exists():
        metadata['n_nifti_series'] = sum([len(files) for _, _, files in os.walk(ds.tmp_nifti_series.value)])
        for _, _, files in os.walk(ds.tmp_nifti_series.value):
            for file in files:
                n_dixon_series += 1 if 'dixon' in file.lower() else 0
        metadata['n_dixon_files'] = n_dixon_series
        nifti_series = os.listdir(ds.tmp_nifti_series.value)[0]
        file_name = ds.tmp_nifti_series.path(nifti_series)
        descrip = nib.load(file_name).header.structarr['descrip']
        metadata['biobank_id'] = json.loads(descrip.tostring().decode('ascii').replace('\x00', ''))['biobank']
        metadata['timestamp'] = time.strftime('%d/%m/%Y %X', time.gmtime(os.path.getmtime(file_name)))
    else:
        metadata['n_nifti_series'] = 0
        metadata['n_dixon_files'] = 0
        metadata['biobank_id'] = ''
        metadata['timestamp'] = ''

    if ds.tmp_nifti_series_donotuse.exists():
        metadata['n_nifti_series_donotuse'] = \
            sum([len(files) for _, _, files in os.walk(ds.tmp_nifti_series_donotuse.value)])
    else:
        metadata['n_nifti_series_donotuse'] = 0

    if not ds.summary.exists():
        os.mkdir(ds.summary.value)
    with open(ds.summary.path('metadata.json'), 'w') as jsonfile:
        json.dump(metadata, jsonfile, sort_keys=True, indent=4, ensure_ascii=False)
