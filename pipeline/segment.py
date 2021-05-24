import os
import shutil
from typing import Optional

import SimpleITK as sitk
import numpy as np
from skimage import measure, morphology

from pipeline.constants import DirectoryStructure as ds, FileNames as fn
from pipeline.logger import log
from pipeline.util import sitk_largest_connected_components, sitk_bbox, sitk_extract_slice, convert_mm_to_voxels, \
    largest_connected_components


def create_body_mask(n_otsu: int = 4, processed: bool = True) -> None:
    dir = ds.nifti if processed else ds.tmp_unprocessed
    vol_ip = sitk.ReadImage(dir.path(fn.ip.value))
    vol_mask_initial = sitk.ReadImage(dir.path(fn.mask.value))

    log.info('Calculating fat and water percentages')
    vol_ip = sitk.Mask(vol_ip, vol_mask_initial)
    vol_ip_otsu = sitk.OtsuMultipleThresholds(sitk.CurvatureFlow(vol_ip, numberOfIterations=8),
                                              numberOfThresholds=n_otsu, valleyEmphasis=True)
    if not os.path.exists(ds.tmp.value):
        os.makedirs(ds.tmp.value)
    sitk.WriteImage(vol_ip_otsu, ds.tmp.path('ip.otsu.nii.gz'))

    vol_fat = sitk.ReadImage(dir.path(fn.fat.value))
    vol_water = sitk.ReadImage(dir.path(fn.water.value))
    vol_fat_percent = sitk.Threshold(sitk.Mask(vol_fat / (vol_fat + vol_water), vol_ip_otsu))
    vol_water_percent = sitk.Threshold(sitk.Mask(vol_water / (vol_fat + vol_water), vol_ip_otsu))

    log.info('Creating binary mask for the whole body')
    vol_ip_percent = sitk.Mask(vol_fat_percent + vol_water_percent, vol_mask_initial)
    x, y, z = vol_ip_percent.GetSize()
    spacing = vol_ip_percent.GetSpacing()
    vol_mask_np = np.zeros((z, y, x), dtype='uint8')

    for s in range(z):
        img_mask_initial = sitk.Extract(vol_mask_initial, (x, y, 0), (0, 0, s))
        if np.sum(sitk.GetArrayFromImage(img_mask_initial)) < 64:
            continue
        img_ip_percent = sitk.Mask(sitk.Extract(vol_ip_percent, (x, y, 0), (0, 0, s)),
                                   sitk.BinaryErode(img_mask_initial, [3]*img_mask_initial.GetDimension()))
        img_add = sitk.ConnectedThreshold(img_ip_percent, [(1, 1)], upper=0.0) \
                  + sitk.ConnectedThreshold(img_ip_percent, [(x - 2, y - 2)], upper=0.0)
        img_mask = img_add <= 0
        if np.sum(sitk.GetArrayFromImage(img_mask)) < 1:
            continue
        vol_mask_np[s, :, :] = sitk.GetArrayFromImage(img_mask)

    vol_mask = sitk.GetImageFromArray(largest_connected_components(vol_mask_np).astype('uint8'))
    vol_mask.CopyInformation(vol_ip_percent)
    vol_mask = sitk.BinaryMorphologicalClosing(vol_mask, convert_mm_to_voxels((2.5, 2.5, 6), spacing))

    if processed:
        if not os.path.exists(ds.analysis.value):
            os.makedirs(ds.analysis.value)
    dir = ds.analysis if processed else ds.tmp_unprocessed
    sitk.WriteImage(vol_fat_percent, dir.path(fn.fat_percent.value))
    sitk.WriteImage(vol_water_percent, dir.path(fn.water_percent.value))
    sitk.WriteImage(vol_mask, dir.path(fn.body_mask.value))


def crop_body_mask(processed: bool = True, number_of_slices: int = 3) -> None:
    log.info(f'Cropping the body mask, top/bottom {number_of_slices} slices')
    if processed:
        dir = ds.analysis
    else:
        dir = ds.tmp_unprocessed
    file_name = dir.path(fn.body_mask.value)
    mask = sitk.ReadImage(file_name)
    shutil.move(file_name, ds.tmp.path(fn.body_mask.value.replace('.nii', '_original.nii')))
    x, y, z = mask.GetSize()
    zero_slices = sitk.Image(x, y, number_of_slices, mask.GetPixelID())
    mask = sitk.Paste(mask, zero_slices, zero_slices.GetSize(), destinationIndex=[0, 0, 0])
    mask = sitk.Paste(mask, zero_slices, zero_slices.GetSize(), destinationIndex=[0, 0, z - number_of_slices])
    sitk.WriteImage(mask, dir.path(fn.body_mask.value))


def create_left_right_mask(processed=True) -> None:
    mask = sitk.ReadImage(ds.nifti.path(fn.mask.value) if processed else ds.tmp_unprocessed.path(fn.mask.value))
    x, _, _ = mask.GetSize()
    boundary, _ = leg_boundary(mask)
    if not processed:
        sitk.WriteImage(boundary, ds.tmp_unprocessed.path('leg_boundary.nii.gz'))
    mask_left_right = {
        'left': sitk.ConnectedThreshold(boundary, [(x - 1, 1, 1)], upper=0.0),
        'right': sitk.ConnectedThreshold(boundary, [(1, 1, 1)], upper=0.0, replaceValue=2)
    }
    mask_left_right['right'] = mask_left_right['right'] + 2 * (boundary > 0)
    mask_left_right = mask_left_right['left'] + mask_left_right['right']
    if processed:
        if not ds.analysis.exists():
            os.makedirs(ds.analysis.value)
        sitk.WriteImage(mask_left_right, ds.analysis.path(fn.left_right_mask.value))
        if ds.analysis.exists(fn.body_mask.value):
            sitk.WriteImage(sitk.ReadImage(ds.analysis.path(fn.body_mask.value)) * mask_left_right,
                            ds.analysis.path(fn.left_right_body_mask.value))
    else:
        sitk.WriteImage(mask_left_right, ds.tmp_unprocessed.path(fn.left_right_mask.value))
        if ds.tmp_unprocessed.exists(fn.body_mask.value):
            sitk.WriteImage(sitk.ReadImage(ds.tmp_unprocessed.path(fn.body_mask.value)) * mask_left_right,
                            ds.tmp_unprocessed.path(fn.left_right_body_mask.value))


def leg_boundary(mask: sitk.Image, start: Optional[int] = None) -> (sitk.Image, int):
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    x, y, z = mask.GetSize()
    spacing = mask.GetSpacing()

    img_midpoint_np = np.zeros((z, x), np.int)
    midpoint = midpoint_previous = start
    for s in range(z):
        img_mask = sitk.BinaryErode(sitk.Extract(mask, [x, y, 0], [0, 0, int(s)]),
                                    convert_mm_to_voxels((2 * 2.5, 4 * 2.5), spacing[:2]),
                                    sitk.sitkBall, 0.0, 1.0, False)
        if np.sum(sitk.GetArrayFromImage(img_mask)) == 0:
            continue
        diff_rowsgt0 = np.diff((sitk.GetArrayFromImage(img_mask).sum(axis=0) > 0).astype('int'))
        if np.sum(diff_rowsgt0 > 0) == 1:
            img_mask = sitk.BinaryErode(sitk.Extract(mask, [x, y, 0], [0, 0, int(s)]),
                                        convert_mm_to_voxels((4 * 2.5, 8 * 2.5), spacing[:2]),
                                        sitk.sitkBall, 0.0, 1.0, False)
            diff_rowsgt0 = np.diff((sitk.GetArrayFromImage(img_mask).sum(axis=0) > 0).astype('int'))
            if np.sum(diff_rowsgt0 > 0) == 1:
                img_mask = sitk.BinaryErode(sitk.Extract(mask, [x, y, 0], [0, 0, int(s)]),
                                            convert_mm_to_voxels((8 * 2.5, 16 * 2.5), spacing[:2]),
                                            sitk.sitkBall, 0.0, 1.0, False)
                diff_rowsgt0 = np.diff((sitk.GetArrayFromImage(img_mask).sum(axis=0) > 0).astype('int'))
                if np.sum(diff_rowsgt0 > 0) == 1:
                    img_mask = sitk.BinaryErode(sitk.Extract(mask, [x, y, 0], [0, 0, int(s)]),
                                                convert_mm_to_voxels((16 * 2.5, 32 * 2.5), spacing[:2]),
                                                sitk.sitkBall, 0.0, 1.0, False)
                    diff_rowsgt0 = np.diff((sitk.GetArrayFromImage(img_mask).sum(axis=0) > 0).astype('int'))
        if np.sum(diff_rowsgt0 > 0) == 2:
            label_mask = measure.label(sitk.GetArrayFromImage(img_mask), connectivity=1)
            props_label_mask = measure.regionprops(label_mask)
            midpoint_new = (props_label_mask[0].centroid[1] + props_label_mask[1].centroid[1]) // 2
            if not midpoint:
                midpoint = int(midpoint_new)
            else:
                if midpoint_new > midpoint_previous:
                    midpoint += 1
                elif midpoint_new < midpoint_previous:
                    midpoint -= 1
            img_midpoint_np[s, midpoint] = 16
            midpoint_previous = midpoint
        else:
            log.debug(f'Leg boundary terminated at slice {s}')
            break
    img_midpoint_np[np.arange(s, z), midpoint] = 16
    vol_midpoint_np = np.stack((img_midpoint_np,) * y, axis=1)
    vol_midpoint_sitk = sitk.GetImageFromArray(vol_midpoint_np.astype('uint8'))
    vol_midpoint_sitk.CopyInformation(mask)

    return vol_midpoint_sitk, midpoint


def segment_visceral_fat(segmentation_directory: str, file_prefix: str = 'otsu_prob_argmax_',
                         fat_threshold: float = 0.5) -> None:
    if os.path.exists(os.path.join(segmentation_directory, f'{file_prefix}abdominal_cavity.nii.gz')):
        abdominal_cavity = sitk.ReadImage(os.path.join(segmentation_directory, f'{file_prefix}abdominal_cavity.nii.gz'))
    else:
        log.error('Visceral fat segmentation failed, abdominal cavity not segmented!')
        return
    abdominal_cavity = sitk.Cast(abdominal_cavity, sitk.sitkUInt16)

    log.info('Visceral fat identified by isolating the abdominal cavity')
    fat_percent = sitk.Mask(sitk.ReadImage(ds.analysis.path(fn.fat_percent.value)),
                            sitk.ReadImage(ds.analysis.path(fn.body_mask.value)))

    try:
        visceral_fat = sitk.Mask(abdominal_cavity, fat_percent > fat_threshold)
    except:
        log.warning('Segmentations shifted')
        origin_fat_percent = fat_percent.GetOrigin()
        abdominal_cavity.SetOrigin(origin_fat_percent)
        visceral_fat = sitk.Mask(abdominal_cavity, fat_percent > fat_threshold)

    log.debug('Identify and remove voxels with small signal intensity in the in-phase volume')
    ip = sitk.Mask(sitk.ReadImage(ds.nifti.path(fn.ip.value)), abdominal_cavity)
    ip10 = np.percentile(sitk.GetArrayFromImage(ip)[sitk.GetArrayFromImage(visceral_fat) > 0], 10)
    log.debug(f'10th percentile of the in-phase signal (abdominal cavity only) = {ip10:4.2f}')
    visceral_fat = sitk.MaskNegated(visceral_fat, sitk.Cast(ip <= ip10, sitk.sitkUInt16))

    log.debug('Removing smallest components from visceral fat')
    visceral_fat_rsc = morphology.remove_small_objects(sitk.GetArrayFromImage(visceral_fat), min_size=2 ** 10)
    visceral_fat = sitk.GetImageFromArray(visceral_fat_rsc)
    visceral_fat.CopyInformation(abdominal_cavity)
    visceral_fat = sitk.VotingBinaryIterativeHoleFilling(visceral_fat, (1, 1, 1), maximumNumberOfIterations=5)

    log.debug('Exclude tissue associated with abdominal organs')
    exclude_organs = {
        'Liver': {'mask': f'{file_prefix}liver.nii.gz'},
        'Spleen': {'mask': f'{file_prefix}spleen.nii.gz'},
        'Left Kidney': {'mask': f'{file_prefix}kidney_left.nii.gz'},
        'Right Kidney': {'mask': f'{file_prefix}kidney_right.nii.gz'},
    }
    for organ, fns in exclude_organs.items():
        if os.path.exists(os.path.join(segmentation_directory, fns['mask'])):
            exclude = sitk.ReadImage(os.path.join(segmentation_directory, fns['mask']))
        else:
            log.error(f'{organ} has not been segmented, visceral fat mask failed...')
            return
        if exclude:
            log.info(f'{organ} removed from the visceral fat mask')
            try:
                visceral_fat = sitk.MaskNegated(visceral_fat, sitk.Cast(exclude, sitk.sitkUInt16))
            except:
                log.info('Segmentations shifted')
                exclude.SetOrigin(fat_percent.GetOrigin())
                visceral_fat = sitk.MaskNegated(visceral_fat, sitk.Cast(exclude, sitk.sitkUInt16))
    sitk.WriteImage(visceral_fat, os.path.join(segmentation_directory, fn.visceral_fat_mask.value))


def segment_abdominal_subcutaneous_fat(segmentation_directory: str, file_prefix: str = 'otsu_prob_argmax_',
                                       fat_threshold: float = 0.7) -> None:
    if os.path.exists(os.path.join(segmentation_directory, f'{file_prefix}body_cavity.nii.gz')):
        body_cavity = sitk.ReadImage(os.path.join(segmentation_directory, f'{file_prefix}body_cavity.nii.gz'))
    else:
        log.error('Abdominal subcutaneous fat segmentation failed, body cavity not segmented!')
        return

    if os.path.exists(os.path.join(segmentation_directory, f'{file_prefix}abdominal_cavity.nii.gz')):
        abdominal_cavity = sitk.ReadImage(os.path.join(segmentation_directory, f'{file_prefix}abdominal_cavity.nii.gz'))
    else:
        log.error('Abdominal subcutaneous fat segmentation failed, abdominal cavity not segmented!')
        return

    log.info('Abdominal subcutaneous fat identified by excluding the body cavity')
    body_mask = sitk.ReadImage(ds.analysis.path(fn.body_mask.value))
    try:
        subcutaneous_fat = sitk.MaskNegated(body_mask, sitk.Cast(body_cavity, sitk.sitkUInt8))
    except:
        log.warning('Segmentations shifted')
        body_cavity.SetOrigin(body_mask.GetOrigin())
        subcutaneous_fat = sitk.MaskNegated(body_mask, sitk.Cast(body_cavity, sitk.sitkUInt8))

    fat_percent = sitk.Mask(sitk.ReadImage(ds.analysis.path(fn.fat_percent.value)), body_mask)
    subcutaneous_fat = sitk.VotingBinaryIterativeHoleFilling(sitk.Mask(subcutaneous_fat, fat_percent > fat_threshold),
                                                             (1, 1, 1), maximumNumberOfIterations=5)
    x, y, z = subcutaneous_fat.GetSize()
    abdominal_mask = sitk.Image(x, y, z, subcutaneous_fat.GetPixelID())
    abdominal_mask.CopyInformation(subcutaneous_fat)
    _, _, z_min, _, _, z_max = sitk_bbox(abdominal_cavity)
    vol_zero = sitk.Image(x, y, int(z_max - z_min), subcutaneous_fat.GetPixelID())
    abdominal_mask = sitk.Paste(abdominal_mask, vol_zero + 1, vol_zero.GetSize(), destinationIndex=[0, 0, int(z_min)])
    abdominal_subcutaneous_fat = sitk.Mask(abdominal_mask, subcutaneous_fat)
    abdominal_subcutaneous_fat = sitk_largest_connected_components(abdominal_subcutaneous_fat)

    sitk.WriteImage(abdominal_subcutaneous_fat,
                    os.path.join(segmentation_directory, fn.abdominal_subcutaneous_fat_mask.value))


def segment_multiecho_pancreas(organ_mask: str) -> None:
    if os.path.exists(organ_mask):
        vol, seg = sitk_extract_slice(sitk.ReadImage(ds.nifti.path(fn.pancreas_norm.value)),
                                      sitk.ReadImage(organ_mask),
                                      sitk.ReadImage(ds.nifti.path(fn.pancreas_mag.value)))
        sitk.WriteImage(vol, ds.analysis.path(fn.multiecho_pancreas_t1w.value))
        sitk.WriteImage(seg, ds.analysis.path(fn.multiecho_pancreas_mask.value))
    else:
        log.error('Mask for the pancreas is not found for this subject')


def extract_organ(organ: str, single_slice: str, segmentation_directory: str,
                  file_prefix: str = 'otsu_prob_argmax_') -> None:
    if organ in ['liver', 'spleen', 'kidney_left', 'kidney_right']:
        ref_vol_3d_file_name = fn.water.value
        if not ds.nifti.exists(ref_vol_3d_file_name):
            log.error(f"Reference volume '{single_slice}' not found")
            return
        ref_vol_3d_in_2d_file_name = f'water.{organ}_from_{single_slice}.nii.gz'
        mask_2d_file_name = f'mask.{organ}_from_{single_slice}'
    elif organ == 'body':
        ref_vol_3d_file_name = fn.ip.value
        if not ds.nifti.exists(ref_vol_3d_file_name):
            log.error(f"Reference volume '{ref_vol_3d_file_name}' not found")
            return
        ref_vol_3d_in_2d_file_name = f'ip.{organ}_from_{single_slice}.nii.gz'
        mask_2d_file_name = f'mask.{organ}_from_{single_slice}.nii.gz'
    else:
        log.error("Only 'liver', 'spleen', 'kidney_left' 'kidney_right' organs")
        return
    if single_slice in ['multiecho_pancreas', 'multiecho_liver', 'ideal_liver']:
        ref_vol_2d_file_name = f'{single_slice}_magnitude.nii.gz'
        if not ds.nifti.exists(ref_vol_2d_file_name):
            log.error(f"Single-slice file '{single_slice}' not found")
            return
    else:
        log.error("Only 'multiecho_pancreas', 'multiecho_liver' or 'ideal_liver' single-slice files")
        return
    if organ == 'body':
        mask_3d_file_name = ds.analysis.path(fn.body_mask.value)
    else:
        mask_3d_file_name = os.path.join(segmentation_directory, f'{file_prefix}{organ}.nii.gz')
    if not os.path.exists(mask_3d_file_name):
        log.error(f"Segmentation file '{mask_3d_file_name}' not found")
        return

    vol, seg = sitk_extract_slice(sitk.ReadImage(ds.nifti.path(ref_vol_3d_file_name)),
                                  sitk.ReadImage(mask_3d_file_name),
                                  sitk.ReadImage(ds.nifti.path(ref_vol_2d_file_name)))
    if organ == 'body':
        seg = sitk_largest_connected_components(seg)
    sitk.WriteImage(vol, os.path.join(segmentation_directory, ref_vol_3d_in_2d_file_name))
    sitk.WriteImage(seg, os.path.join(segmentation_directory, mask_2d_file_name))
