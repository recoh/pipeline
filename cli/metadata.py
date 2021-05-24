import os

from pipeline.metadata import subject_metadata


class Metadata(object):

    def generate(self, subject_directory):
        os.chdir(subject_directory)
        subject_metadata()
