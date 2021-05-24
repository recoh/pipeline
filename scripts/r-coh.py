#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import shutil
import time

import fire

from cli.bone_joints import BoneJoints
from cli.dicom import Dicom
from cli.metadata import Metadata
from cli.multiecho import Multiecho
from cli.nifti import Nifti
from cli.quality_control import QC
from cli.segment import Segment
from pipeline.constants import DirectoryStructure as ds
from pipeline.logger import LOG_FILE, log


class Pipeline(object):

    def __init__(self):
        self.dicom = Dicom()
        self.nifti = Nifti()
        self.metadata = Metadata()
        self.qc = QC()
        self.segment = Segment()
        self.multiecho = Multiecho()
        self.bone_joints = BoneJoints()

    def process(self, input_directory: str, output_directory: str, biobank_project: str,
                skip_bias_correction: bool = False, skip_swap_correction: bool = False):
        """
        Process UK Biobank DICOM data from the abdominal MRI acquisition protocol.

        :param input_directory: Full path to the directory of DICOM data, single subject only.
        :param output_directory: Full path for the single-subject processed data to be written.
        :param biobank_project: The project ID from the UK Biobank for these data.
        :param skip_bias_correction: Skip bias field correction (default = False).
        :param skip_swap_correction: Skip fat-water swap correction (default = False).
        """

        input_directory = os.path.realpath(input_directory)
        output_directory = os.path.realpath(output_directory)
        if biobank_project:
            biobank_project = f'{biobank_project:>05}'
        self.dicom.organize(input_directory, output_directory, biobank_project)
        dicom_id, biobank_id = self.dicom.convert(input_directory, output_directory, biobank_project)
        log.info(f'DICOM ID:\t{dicom_id}')
        if biobank_project:
            log.info(f'Biobank Project:\t{biobank_project}')
            log.info(f'Biobank ID:\t{biobank_id}')
        os.chdir(os.path.join(output_directory, dicom_id))
        self.nifti.assemble(ds.root.value, skip_bias_correction, skip_swap_correction)
        self.metadata.generate(ds.root.value)
        self.segment.body_mask(ds.root.value, biobank=biobank_project)
        self.segment.left_right_mask(ds.root.value)
        self.qc.anomaly_detection(ds.root.value)
        self.bone_joints.predict(ds.root.value)
        self.nifti.copy_liver_and_pancreas(ds.root.value)
        self.multiecho.pdff(ds.root.value, 'multiecho_liver')
        self.multiecho.pdff(ds.root.value, 'multiecho_pancreas')
        self.multiecho.pdff(ds.root.value, 'ideal_liver')
        self.multiecho.iron(ds.root.value, 'multiecho_liver')
        self.multiecho.iron(ds.root.value, 'multiecho_pancreas')
        self.multiecho.iron(ds.root.value, 'ideal_liver')


def main():
    start_time = time.time()
    try:
        result = fire.Fire(Pipeline)
    except SystemExit as e:
        print(f'SystemExit: {e}')
    except ValueError as e:
        print(f'ValueError: {e}')
    else:
        if not result:
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
            log.info(f'Processing completed in {elapsed_time}')
            logging.shutdown()
            if not ds.summary_logs.exists():
                os.makedirs(ds.summary_logs.value)
            file_name = f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime(start_time))}.log"
            shutil.copyfile(LOG_FILE.name, ds.summary_logs.path(file_name))


if __name__ == '__main__':
    main()
