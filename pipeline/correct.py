from typing import Dict, Tuple

import SimpleITK as sitk
import numpy as np
from dcmstack import NiftiWrapper

from pipeline.constants import DataChannel as dc
from pipeline.logger import log
from pipeline.nifti_series import add_series_to_dict, replace_nii_img_dataobj
from pipeline.util import nibabel_to_sitk, sitk_to_nibabel, channel_to_str


def estimate_bias_field(input_volume: NiftiWrapper, mask_volume: NiftiWrapper, fwhm: float = 0.15,
                        iterations: Tuple[int, int, int, int] = (50, 50, 50, 50)) -> (np.ndarray, np.ndarray):
    # sitk_input_volume = nibabel_to_sitk(input_volume.nii_img)
    # sitk_mask_volume = nibabel_to_sitk(mask_volume.nii_img)
    # input_volume_shrink = sitk.Shrink(sitk_input_volume, [2] * sitk_input_volume.GetDimension())
    # mask_volume_shrink = sitk.Shrink(sitk_mask_volume, [2] * sitk_mask_volume.GetDimension())
    #
    # corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # corrector.SetMaximumNumberOfIterations(iterations)
    # output = corrector.Execute(sitk.Cast(input_volume_shrink, sitk.sitkFloat32),
    #                            sitk.Cast(mask_volume_shrink, sitk.sitkUInt8))
    # log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_input_volume)
    # bias_field = sitk.Exp(log_bias_field)
    # corrected = np.asanyarray(sitk_to_nibabel(sitk_input_volume / bias_field).dataobj)
    #
    # return bias_field, corrected

    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetBiasFieldFullWidthAtHalfMaximum(fwhm)  # default = 0.15
    n4.SetMaximumNumberOfIterations(iterations)  # default = 50 x 50 x 50 x 50

    sitk_input_volume = nibabel_to_sitk(input_volume.nii_img)
    sitk_mask_volume = nibabel_to_sitk(mask_volume.nii_img)
    sitk_corrected = n4.Execute(sitk.Cast(sitk_input_volume, sitk.sitkFloat32),
                                sitk.Cast(sitk_mask_volume, sitk.sitkUInt8))
    corrected = np.asanyarray(sitk_to_nibabel(sitk_corrected).dataobj)
    bias_field = np.where(corrected > 0, np.asanyarray(input_volume.nii_img.dataobj) / corrected, 0)

    return bias_field, corrected


def bias_field_correction(series_data: Dict, fwhm: float = 0.15, iterations: Tuple = (10, 10, 10, 10)) -> None:
    in_phase_series = series_data[dc.in_phase]
    mask_series = series_data[dc.mask]

    if not in_phase_series or len(in_phase_series) < 1:
        log.error(f"No '{channel_to_str(dc.in_phase)}' series!")
        return

    log.debug(f'Bias field correction for {len(in_phase_series)} series')

    n4bias_series = dict()
    for location, vol in in_phase_series.items():
        mask = mask_series[location if location != 0 else 0.0]
        field, corrected = estimate_bias_field(vol, mask, fwhm, iterations)
        n4bias_series[location] = field
        series_data = add_series_to_dict(replace_nii_img_dataobj(vol, corrected.astype('uint16')),
                                         series_data, dc.in_phase, location)

    for channel in [dc.opp_phase, dc.water, dc.fat]:
        for location, vol in series_data[channel].items():
            log.debug(f'Correcting volume at {location:4.2f}')
            corrected = np.where(n4bias_series[location] > 0,
                                 np.asanyarray(vol.nii_img.dataobj) / n4bias_series[location], 0)
            series_data = add_series_to_dict(replace_nii_img_dataobj(vol, corrected.astype('uint16')),
                                             series_data, channel, location)
