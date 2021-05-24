import copy
import os
from typing import Dict, Optional

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from dcmstack import NiftiWrapper
from nibabel.affines import apply_affine
from scipy import ndimage as nd
from skimage import filters, morphology

from pipeline.constants import DataChannel as dc, DirectoryStructure as ds, NIFTI_EXT, PLOT_EXT
from pipeline.logger import log
from pipeline.plot import coronal_view, rescale
from pipeline.segment import leg_boundary
from pipeline.util import nibabel_to_sitk, sitk_to_nibabel, channel_to_str, largest_connected_components

REFERENCE_ORIGIN = [-250.0, 191.96429443359375, -462.75]
REFERENCE_SIZE = [224, 174, 44]


def resample_series_y(series_data: Dict, target_y: int = 174):
    log.debug('Resampling volumes to Y = {}'.format(target_y))

    x, y = REFERENCE_SIZE[:2]

    for channel in series_data.keys():
        if channel == dc.mask:
            interpolation = sitk.sitkNearestNeighbor
        else:
            interpolation = sitk.sitkBSpline
        for location, volume in series_data[channel].items():
            if volume.nii_img.shape[1] == target_y:
                continue

            sitk_volume = nibabel_to_sitk(volume.nii_img)
            new_sitk_volume = sitk.Image(x, y, sitk_volume.GetSize()[2], sitk.sitkInt16)
            new_origin = copy.copy(REFERENCE_ORIGIN)
            new_origin[2] = sitk_volume.GetOrigin()[2]
            new_sitk_volume.SetOrigin(new_origin)
            new_sitk_volume.SetSpacing(sitk_volume.GetSpacing())
            new_sitk_volume.SetDirection(sitk_volume.GetDirection())
            new_sitk_volume = sitk.Resample(sitk_volume, new_sitk_volume, sitk.Euler3DTransform(), interpolation)
            if interpolation == sitk.sitkBSpline:
                new_sitk_volume = sitk.Mask(new_sitk_volume, new_sitk_volume >= 0)

            new_volume = sitk_to_nibabel(new_sitk_volume)
            series_data = add_series_to_dict(replace_nii_img_nifti1image(volume, new_volume), series_data, channel,
                                             location)


def resample_series_z(series_data: Dict, target_z: float = 3.0):
    log.debug('Resampling volumes to SliceThickness = {}'.format(target_z))

    for channel in series_data.keys():
        if channel == dc.mask:
            interpolation = sitk.sitkNearestNeighbor
        else:
            interpolation = sitk.sitkBSpline
        for location, volume in series_data[channel].items():
            if volume.nii_img.header.get_zooms()[2] == target_z:
                continue

            sitk_volume = nibabel_to_sitk(volume.nii_img)
            new_spacing = np.asanyarray(sitk_volume.GetSpacing(), dtype='float32')
            new_spacing[2] = target_z

            spacing = np.asanyarray(sitk_volume.GetSpacing(), dtype='float32')
            size = np.asanyarray(sitk_volume.GetSize(), dtype='float32')
            new_size = np.maximum((1, 1, 1), np.round(size * spacing / new_spacing)).astype('uint16')
            new_volume = sitk.Image(new_size.tolist(), sitk.sitkFloat32)
            new_volume.SetOrigin(sitk_volume.GetOrigin())
            new_volume.SetDirection(sitk_volume.GetDirection())
            new_volume.SetSpacing(new_spacing.tolist())
            new_sitk_volume = sitk.Resample(sitk_volume, new_volume, sitk.Euler3DTransform(), interpolation)
            if interpolation == sitk.sitkBSpline:
                new_sitk_volume = sitk.Mask(new_sitk_volume, new_sitk_volume >= 0)

            new_volume = sitk_to_nibabel(new_sitk_volume)
            series_data = add_series_to_dict(replace_nii_img_nifti1image(volume, new_volume), series_data, channel,
                                             location)


def blend_series_by_channel(series_data: Dict, size: int = 5, padding: int = 8) -> Dict:
    volumes = dict()
    for channel, series in series_data.items():
        blend = assemble_series(channel, series)
        if channel == dc.mask:
            padded = np.pad(np.asanyarray(blend.nii_img.dataobj), [(padding, padding)] * 3, mode='edge')
            padded = nd.binary_closing(padded, morphology.ball(size))
            padded = nd.binary_opening(padded, morphology.ball(size)).astype('uint8')
            padded = largest_connected_components(padded)
            blend = replace_nii_img_dataobj(blend, padded[padding:-padding, padding:-padding, padding:-padding])
        volumes = add_series_to_dict(blend, volumes, channel, 0.0)

    return volumes


def assemble_series(channel: dc, series: Dict[float, NiftiWrapper], debug: bool = False) -> NiftiWrapper:
    log.info('Assembling {}'.format(channel_to_str(channel)))
    if not series or len(series) == 0:
        log.error('No series to assemble')
        raise RuntimeError

    blend = overlap_in_slices = prev_volume = None
    for location, volume in sorted(series.items()):
        log.debug('Location = {:4.2f}'.format(location))
        if blend is not None:
            z2 = prev_volume.nii_img.dataobj.shape[2]
            min_slice_loc1 = apply_affine(volume.nii_img.affine, [[0, 0, 1]])[0][2]
            max_slice_loc2 = apply_affine(prev_volume.nii_img.affine, [[0, 0, z2]])[0][2]
            if debug:
                log.debug('Slice location (min_1, max_2): {:4.2f}, {:4.2f}'.format(min_slice_loc1, max_slice_loc2))
            slice_thickness = volume.nii_img.header.get_zooms()[2]
            overlap_in_slices = ((max_slice_loc2 - min_slice_loc1) / slice_thickness) + 1
            if overlap_in_slices < 0:
                log.error('No overlap between slices: overlap = {}'.format(overlap_in_slices))
                raise RuntimeError
            overlap_in_slices = int(round(overlap_in_slices, 0))
            if debug:
                log.debug('Blending with {}-slice overlap ({:4.2f}mm)'.format(overlap_in_slices,
                                                                              overlap_in_slices * slice_thickness))

        blend = blend_two_series(blend, volume, overlap_in_slices)
        prev_volume = volume

    blend.nii_img.header.structarr['descrip'] = volume.nii_img.header.structarr['descrip']

    return blend


def blend_two_series(series1: NiftiWrapper, series2: NiftiWrapper, overlap: int) -> NiftiWrapper:
    if not series1:
        return series2
    if not series2:
        return series1
    if overlap < 0:
        log.error('Negative overlap ({} slices)'.format(overlap))
        raise RuntimeError

    affine1 = series1.nii_img.affine.copy()
    affine2 = series2.nii_img.affine.copy()
    min_z = min(affine1[2, 3], affine2[2, 3])
    affine1[2, 3] = affine2[2, 3] = min_z
    if (not np.allclose(affine1[:3, :3], affine2[:3, :3], atol=1e-3)
            and not np.allclose(affine1[:, 3], affine2[:, 3], atol=1e-3)):
        log.error('{} != {}'.format(affine1, affine2))
        raise RuntimeError

    data1 = np.asanyarray(series1.nii_img.dataobj)
    data2 = np.asanyarray(series2.nii_img.dataobj)
    shape1 = data1.shape
    shape2 = data2.shape
    if shape1[0] != shape2[0]:
        log.error('X-dimension: {} != {}'.format(shape1[0], shape2[0]))
        raise RuntimeError
    if shape1[1] != shape2[1]:
        log.error('Y-dimension: {} != {}'.format(shape1[1], shape2[1]))
        raise RuntimeError

    header1 = series1.nii_img.header
    if overlap == 0:
        log.debug('No overlap, stacking volumes')
        new_series = np.dstack((data1, data2))
        return NiftiWrapper(nib.Nifti1Image(new_series, affine1, header1))

    weights = np.array(1 / (1 + np.exp(np.linspace(-5, 5, overlap))))  # Logistic function
    blend = np.zeros((shape1[0], shape1[1], overlap), dtype='float32')
    for z in range(overlap):
        blend[:, :, z] = data1[:, :, z - overlap] * weights[z] + data2[:, :, z] * (1 - weights[z])
    new_series = np.dstack((data1[:, :, :-overlap], blend.astype('uint16'), data2[:, :, overlap:]))

    return NiftiWrapper(nib.Nifti1Image(new_series, affine1, header1))


def add_series_to_dict(series: NiftiWrapper, series_data: Dict, channel: dc = None,
                       location: float = None) -> Optional[Dict]:
    manufacturer = series.get_meta('Manufacturer').lower().strip()
    series_description = series.get_meta('SeriesDescription').lower().strip()
    if channel is None:
        if manufacturer == 'siemens':
            if series_description.find('dixon') >= 0:
                if series_description.endswith('_w'):
                    channel = dc.water
                elif series_description.endswith('_f'):
                    channel = dc.fat
                elif series_description.endswith('_in'):
                    channel = dc.in_phase
                elif series_description.endswith('_opp'):
                    channel = dc.opp_phase
        else:
            log.error("Unknown Manufacturer '{}'".format(manufacturer))
            return

    if channel is None:
        log.error("Unknown channel for SeriesDescription '{}'".format(series_description))
        return

    if channel not in series_data.keys():
        series_data[channel] = dict()
    if not location and location != 0.0:
        location = round(min(series.meta_ext.get_values('SliceLocation')), 3)
    if location in series_data[channel]:
        log.warning('Replacing series at {:4.2f} for {}'.format(location, channel_to_str(channel)))
    series_data[channel][location] = series

    return series_data


def load_dixon_series(input_directory: str) -> Dict:
    series_data = dict()
    for root, _, files in os.walk(input_directory):
        for file in files:
            if 'dixon' in file.lower():
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    file_with_root = os.path.join(root, file)
                    log.debug('Loading series {}'.format(os.path.basename(file_with_root)))
                    volume = NiftiWrapper.from_filename(file_with_root)
                    series_data = add_series_to_dict(volume, series_data)
                else:
                    log.debug('Skipping non-NIfTI file {}'.format(file))
            else:
                log.warning('Skipping non-Dixon file {}'.format(file))

    return series_data


def save_dixon_volumes(volumes: Dict, processed: bool = True) -> None:
    for channel, volume in volumes.items():
        if len(volume) != 1:
            log.error('Expected single blended volume, there are {} instead of {}'.format(len(volume), volume))
        volume = list(volume.values())[0]
        channel_name = channel_to_str(channel)
        if processed:
            if not ds.nifti.exists():
                os.makedirs(ds.nifti.value)
            if not ds.plots_nifti.exists():
                os.makedirs(ds.plots_nifti.value)
            file_name = ds.nifti.path('{}.{}'.format(channel_name, NIFTI_EXT))
            plots_path = ds.plots_nifti.path
        else:
            if not ds.tmp_unprocessed.exists():
                os.makedirs(ds.tmp_unprocessed.value)
            if not ds.tmp_plots.exists():
                os.makedirs(ds.tmp_plots.value)
            file_name = ds.tmp_unprocessed.path('{}.{}'.format(channel_name, NIFTI_EXT))
            plots_path = ds.tmp_plots.path
        log.info('Channel = {}'.format(channel_name))
        volume.to_filename(file_name)
        if dc.mask in volumes.keys():
            mask_view = coronal_view(volume.nii_img)
            mask_view.save(plots_path('{}.{}'.format(channel_name, PLOT_EXT)))
        else:
            coronal_view(volume.nii_img).save(plots_path('{}.{}'.format(channel_name, PLOT_EXT)))


def create_mask_by_series(series_data: Dict, channel: dc = dc.in_phase, size: int = 5, padding: int = 8) -> None:
    log.debug('Creating mask from {}'.format(channel_to_str(channel)))
    for location, volume in sorted(series_data[channel].items()):
        log.debug('Creating mask for volume at {:4.2f}'.format(location))
        data = np.asanyarray(volume.nii_img.dataobj)
        p = np.percentile(data, 97.5)
        data[data > p] = p
        data = nd.median_filter(data, size=size, mode='nearest')
        data_rescale = rescale(data)
        mask = np.zeros(data.shape, dtype='float32')
        for z in range(data.shape[2]):
            global_thresh = filters.threshold_otsu(filters.threshold_local(data_rescale[:, :, z], block_size=size))
            mask[:, :, z] = data_rescale[:, :, z] > global_thresh
        mid_slice = mask.copy()
        mid_slice[:, :, mid_slice.shape[2] // 2] = 1
        mid_slice = largest_connected_components(mid_slice)
        mask[mid_slice == 0] = 0
        mask = np.pad(mask, [(padding, padding)] * 3, mode='symmetric')
        for z in range(mask.shape[2]):
            mask[:, :, z] = nd.binary_fill_holes(nd.binary_dilation(mask[:, :, z], morphology.disk(size)))
        mask = nd.binary_closing(mask, morphology.ball(size))[padding:-padding, padding:-padding, padding:-padding]
        series_data = add_series_to_dict(replace_nii_img_dataobj(volume, mask.astype('uint8')),
                                         series_data, dc.mask, location)


def create_midpoint_by_series(series_data: Dict, channel: dc = dc.mask) -> None:
    log.debug('Creating midpoint from {}'.format(channel_to_str(channel)))
    midpoint_z = None
    for location, volume in sorted(series_data[channel].items()):
        log.debug('Creating midpoint for volume at {:4.2f}'.format(location))
        size = volume.nii_img.dataobj.shape
        boundary, midpoint_z = leg_boundary(nibabel_to_sitk(volume.nii_img), start=midpoint_z)
        mask_left_right = {
            'left': sitk.ConnectedThreshold(boundary, [(size[0] - 1, 1, 1)], upper=0.0),
            'right': sitk.ConnectedThreshold(boundary, [(1, 1, 1)], upper=0.0, replaceValue=2)
        }
        mask_left_right['right'] = mask_left_right['right'] + 2 * (boundary > 0)
        midpoint = sitk_to_nibabel(mask_left_right['left'] + mask_left_right['right'])
        series_data = add_series_to_dict(
            replace_nii_img_dataobj(volume, np.asanyarray(midpoint.dataobj).astype('uint8')),
            series_data, dc.midpoint, location)


def replace_nii_img_dataobj(img: NiftiWrapper, new_img_data: np.ndarray) -> NiftiWrapper:
    return NiftiWrapper(nib.Nifti1Image(new_img_data, img.nii_img.affine.copy(), copy.deepcopy(img.nii_img.header)))


def replace_nii_img_nifti1image(img: NiftiWrapper, new_img: nib.Nifti1Image) -> NiftiWrapper:
    new_img.header.structarr['descrip'] = img.nii_img.header.structarr['descrip']
    new_img.header.extensions = nib.nifti1.Nifti1Extensions([img.meta_ext])

    return NiftiWrapper(nib.Nifti1Image(np.asanyarray(new_img.dataobj), new_img.affine.copy(),
                                        copy.deepcopy(new_img.header)))
