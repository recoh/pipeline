import os

from pipeline.anomaly import anomaly_detection
from pipeline.sanity import slice_location


class QC(object):

    def anomaly_detection(self, subject_directory):
        os.chdir(subject_directory)
        anomaly_detection()

    def slice_location(self, subject_directory, organ, segmentation_file=None):
        os.chdir(subject_directory)
        slice_location(organ, segmentation_file)
