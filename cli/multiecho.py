import os

from cli.segment import Segment
from pipeline.constants import DirectoryStructure as ds
from pipeline.multiecho_pdff import fat_fraction_from_multiecho, iron_concentration_from_r2star


class Multiecho(object):

    def pdff(self, subject_directory, organ, magnitude_only=False):
        os.chdir(subject_directory)
        fat_fraction_from_multiecho(organ, magnitude_only)
        Segment.multiecho_organ(self, ds.root.value, 'body', organ, ds.analysis.value)

    def iron(self, subject_directory, organ, magnitude_only=False):
        os.chdir(subject_directory)
        iron_concentration_from_r2star(organ, magnitude_only)
