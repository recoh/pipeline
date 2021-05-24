import os

from pipeline.plot import plot_body_data
from pipeline.segment import create_body_mask, crop_body_mask, create_left_right_mask, segment_multiecho_pancreas, \
    segment_visceral_fat, segment_abdominal_subcutaneous_fat, extract_organ


class Segment(object):

    def body_mask(self, subject_directory: str, biobank: str = None):
        os.chdir(subject_directory)
        create_body_mask()
        if biobank:
            crop_body_mask()
        plot_body_data()

    def left_right_mask(self, subject_directory: str, processed: bool = True):
        os.chdir(subject_directory)
        create_left_right_mask(processed)

    def multiecho_pancreas(self, subject_directory: str, organ_mask: str):
        os.chdir(subject_directory)
        segment_multiecho_pancreas(organ_mask)

    def visceral_fat(self, subject_directory: str, segmentation_directory: str):
        os.chdir(subject_directory)
        segment_visceral_fat(segmentation_directory)

    def abdominal_subcutaneous_fat(self, subject_directory: str, segmentation_directory: str):
        os.chdir(subject_directory)
        segment_abdominal_subcutaneous_fat(segmentation_directory)

    def multiecho_organ(self, subject_directory: str, organ: str, single_slice: str, segmentation_directory: str):
        os.chdir(subject_directory)
        extract_organ(organ, single_slice, segmentation_directory)
