import os
from enum import Enum

NIFTI_EXT = 'nii.gz'
PLOT_EXT = 'png'
JSON_EXT = 'json'

VALID_BIOBANK_PROJECTS = [
    '00000',  # Default
]


class NoValue(Enum):
    """
    https://docs.python.org/3/library/enum.html
    """

    def __repr__(self):
        return '<{}.{}>'.format(self.__class__.__name__, self.name)


class AutoNumber(NoValue):
    def __new__(cls):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj


class DataChannel(AutoNumber):
    in_phase = ()
    opp_phase = ()
    water = ()
    fat = ()
    mask = ()
    mapping = ()
    midpoint = ()


class DirectoryStructure(Enum):
    root = '.'
    annotations = 'annotations'
    analysis = 'analysis'
    nifti = 'nifti'
    plots = 'plots'
    plots_analysis = 'plots/analysis'
    plots_multiecho = 'plots/multiecho'
    plots_nifti = 'plots/nifti'
    summary = 'summary'
    summary_logs = 'summary/logs'
    tmp = 'tmp'
    tmp_plots = 'tmp/plots'
    tmp_nifti_series = 'tmp/nifti_series'
    tmp_nifti_series_donotuse = 'tmp/nifti_series_donotuse'
    tmp_dicom_series = 'tmp/dicom_series'
    tmp_unprocessed = 'tmp/unprocessed'
    landmarks = 'landmarks'

    def __str__(self):
        return str(self.value)

    def path(self, *paths) -> str:
        return os.path.join(self.value, *paths)

    def exists(self, *paths) -> bool:
        return os.path.exists(self.path(*paths))


class FileNames(Enum):
    # unprocessed/
    ip = '{}.{}'.format('ip', NIFTI_EXT)
    op = '{}.{}'.format('op', NIFTI_EXT)
    fat = '{}.{}'.format('fat', NIFTI_EXT)
    water = '{}.{}'.format('water', NIFTI_EXT)
    mask = 'mask.{}'.format(NIFTI_EXT)

    # nifti/
    pancreas = 't1_vibe_pancreas.{}'.format(NIFTI_EXT)
    pancreas_norm = 't1_vibe_pancreas_norm.{}'.format(NIFTI_EXT)
    liver_mag = 'multiecho_liver_magnitude.{}'.format(NIFTI_EXT)
    liver_phase = 'multiecho_liver_phase.{}'.format(NIFTI_EXT)
    ideal_liver_mag = 'ideal_liver_magnitude.{}'.format(NIFTI_EXT)
    ideal_liver_phase = 'ideal_liver_phase.{}'.format(NIFTI_EXT)
    shmolli_mag = 'shmolli_magnitude.{}'.format(NIFTI_EXT)
    shmolli_phase = 'shmolli_phase.{}'.format(NIFTI_EXT)
    shmolli_t1map = 'shmolli_t1map.{}'.format(NIFTI_EXT)
    shmolli_params = 'shmolli_fitparams.{}'.format(NIFTI_EXT)
    pancreas_mag = 'multiecho_pancreas_magnitude.{}'.format(NIFTI_EXT)
    pancreas_phase = 'multiecho_pancreas_phase.{}'.format(NIFTI_EXT)

    # analysis/
    fat_percent = '{}.percent.{}'.format('fat', NIFTI_EXT)
    water_percent = '{}.percent.{}'.format('water', NIFTI_EXT)
    body_mask = 'mask.body.{}'.format(NIFTI_EXT)
    body_cavity_mask = 'mask.body_cavity.{}'.format(NIFTI_EXT)
    abdominal_cavity_mask = 'mask.abdominal_cavity.{}'.format(NIFTI_EXT)
    abdominal_subcutaneous_fat_mask = 'mask.abdominal_subcutaneous_fat.{}'.format(NIFTI_EXT)
    visceral_fat_mask = 'mask.visceral_fat.{}'.format(NIFTI_EXT)
    leg_boundary = 'mask.leg_boundary.{}'.format(NIFTI_EXT)
    left_right_mask = 'mask.left_right.{}'.format(NIFTI_EXT)
    left_right_body_mask = 'mask.body_left_right.{}'.format(NIFTI_EXT)
    liver_mask = 'mask.liver.{}'.format(NIFTI_EXT)
    pancreas_mask = 'mask.pancreas.{}'.format(NIFTI_EXT)
    multiecho_liver_mask = 'mask.multiecho_liver.{}'.format(NIFTI_EXT)
    ideal_liver_mask = 'mask.ideal_liver.{}'.format(NIFTI_EXT)
    multiecho_pancreas_t1w = 't1w.multiecho_pancreas.{}'.format(NIFTI_EXT)
    multiecho_pancreas_mask = 'mask.multiecho_pancreas.{}'.format(NIFTI_EXT)
    
    # annotations
    bone_joints = 'bone_joints.{}'.format(NIFTI_EXT)
    bone_joints_initial_mask = 'mask.bone_joints_atlas_freq.{}'.format(NIFTI_EXT)
    bone_joints_mask = 'mask.bone_joints.{}'.format(NIFTI_EXT)
    bone_joints_spheres = 'mask.bone_joints_as_spheres.{}'.format(NIFTI_EXT)
    bone_joints_json = 'bone_joints.{}'.format(JSON_EXT)

    # summary/
    metadata = 'metadata.{}'.format(JSON_EXT)

    def __str__(self):
        return str(self.value)
