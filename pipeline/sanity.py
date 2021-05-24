import json
import os

import SimpleITK as sitk

from pipeline.constants import DirectoryStructure as ds, FileNames as fn
from pipeline.logger import log
from pipeline.util import sitk_bbox


def slice_location(organ: str, segmentation_file: str = None) -> None:
    if organ == 'multiecho_liver':
        if ds.nifti.exists(fn.liver_mag.value):
            file_magnitude = ds.nifti.path(fn.liver_mag.value)
        else:
            log.error('No magnitude {} file has been detected'.format(organ))
            return
        if ds.analysis.exists(fn.liver_mask.value):
            log.debug('Using 3D segmentation {}'.format(ds.analysis.path(fn.liver_mask.value)))
            segmentation_file = ds.analysis.path(fn.liver_mask.value)
        elif segmentation_file and os.path.exists(segmentation_file):
            log.debug('Using 3D segmentation {}'.format(segmentation_file))
            segmentation_file = segmentation_file
        elif ds.analysis.exists(fn.liver_mask_atlas.value):
            log.debug('Using 3D segmentation {}'.format(ds.analysis.path(fn.liver_mask_atlas.value)))
            segmentation_file = ds.analysis.path(fn.liver_mask_atlas.value)
        else:
            log.warning('No 3D segmentation {} file has been detected, check segmentation_file parameter?'.format(organ))
    elif organ == 'ideal_liver':
        if ds.nifti.exists(fn.ideal_liver_mag.value):
            file_magnitude = ds.nifti.path(fn.ideal_liver_mag.value)
        else:
            log.error('No magnitude {} file has been detected'.format(organ))
            return
        if ds.analysis.exists(fn.liver_mask.value):
            log.debug('Using 3D segmentation {}'.format(ds.analysis.path(fn.liver_mask.value)))
            segmentation_file = ds.analysis.path(fn.liver_mask.value)
        elif segmentation_file and os.path.exists(segmentation_file):
            log.debug('Using 3D segmentation {}'.format(segmentation_file))
            segmentation_file = segmentation_file
        elif ds.analysis.exists(fn.liver_mask_atlas.value):
            log.debug('Using 3D segmentation {}'.format(ds.analysis.path(fn.liver_mask_atlas.value)))
            segmentation_file = ds.analysis.path(fn.liver_mask_atlas.value)
        else:
            log.warning('No 3D segmentation {} file has been detected, check segmentation_file parameter?'.format(organ))
    elif organ == 'multiecho_pancreas':
        if ds.nifti.exists(fn.pancreas_mag.value):
            file_magnitude = ds.nifti.path(fn.pancreas_mag.value)
        else:
            log.error('No magnitude {} file has been detected'.format(organ))
            return
        if ds.analysis.exists(fn.pancreas_mask.value):
            log.debug('Using 3D segmentation {}'.format(ds.analysis.path(fn.pancreas_mask.value)))
            segmentation_file = ds.analysis.path(fn.pancreas_mask.value)
        elif segmentation_file and os.path.exists(segmentation_file):
            log.debug('Using 3D segmentation {}'.format(segmentation_file))
            segmentation_file = segmentation_file
        elif ds.analysis.exists(fn.pancreas_mask_atlas.value):
            log.debug('Using 3D segmentation {}'.format(ds.analysis.path(fn.pancreas_mask_atlas.value)))
            segmentation_file = ds.analysis.path(fn.pancreas_mask_atlas.value)
        else:
            log.warning('No 3D segmentation {} file has been detected, check segmentation_file parameter?'.format(organ))
    else:
        log.error("Only 'multiecho_liver', 'multiecho_pancreas' and 'ideal_liver' are valid organs")
        return

    metadata = dict()
    if ds.summary.exists(fn.metadata.value):
        with open(ds.summary.path(fn.metadata.value), 'r') as jsonfile:
            metadata = json.load(jsonfile)
    if organ not in metadata.keys() or not isinstance(metadata[organ], dict):
        metadata[organ] = dict()
    metadata[organ]['slice_location'] = dict()

    img_2d = sitk.ReadImage(file_magnitude)
    x, y, z, _ = img_2d.GetSize()
    px, py, pz, _ = img_2d.TransformIndexToPhysicalPoint((x // 2, y // 2, 0, 0))
    log.info('Physical slice location (z) = {:6.2f} for midpoint of the 2D image'.format(pz))
    metadata[organ]['slice_location']['physical'] = pz

    img_3d = sitk.ReadImage(ds.nifti.path(fn.water.value))
    sx, sy, sz = img_3d.TransformPhysicalPointToIndex((px, py, pz))
    log.info('Dixon slice location (z) = {:d}'.format(sz))
    metadata[organ]['slice_location']['dixon'] = sz

    if segmentation_file and os.path.exists(segmentation_file):
        img_3d_seg = sitk.ReadImage(segmentation_file)
        x_min, y_min, z_min, x_max, y_max, z_max = sitk_bbox(img_3d_seg)
        log.debug('min(x) = {} and max(x) = {} for 3D organ'.format(x_min, x_max))
        log.debug('min(y) = {} and max(y) = {} for 3D organ'.format(y_min, y_max))
        log.debug('min(z) = {} and max(z) = {} for 3D organ'.format(z_min, z_max))
        slice_percentile = (sz - z_min) / (z_max - z_min)
        log.info('Slice location relative to 3D segmentation = {:.1f}%'.format(100 * slice_percentile))
        metadata[organ]['slice_location']['percentile_organ'] = slice_percentile
    else:
        log.warning("Cannot find file '{}'".format(segmentation_file))

    log.info('Writing slice locations to {}'.format(fn.metadata.value))
    if not ds.summary.exists():
        os.mkdir(ds.summary.value)
    with open(ds.summary.path(fn.metadata.value), 'w') as jsonfile:
        json.dump(metadata, jsonfile, sort_keys=True, indent=4, ensure_ascii=False)
