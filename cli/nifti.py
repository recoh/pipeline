import os

from pipeline.nifti import assemble_and_correct_series, copy_liver_and_pancreas_series


class Nifti(object):

    def assemble(self, subject_directory, skip_bias_correction=False, skip_swap_correction=False):
        os.chdir(subject_directory)
        assemble_and_correct_series(skip_bias_correction, skip_swap_correction)

    def copy_liver_and_pancreas(self, subject_directory):
        os.chdir(subject_directory)
        copy_liver_and_pancreas_series()
