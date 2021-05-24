import os
import shutil

from dcmstack import NiftiWrapper

from pipeline.constants import DirectoryStructure as ds, FileNames as fn, NIFTI_EXT
from pipeline.correct import bias_field_correction
from pipeline.logger import log
from pipeline.nifti_series import load_dixon_series, resample_series_y, resample_series_z, blend_series_by_channel, \
    save_dixon_volumes, create_mask_by_series
from pipeline.swaps import swap_detection, any_swaps, fix_fat_water_swaps


def assemble_and_correct_series(skip_bias_correction: bool = False, skip_swap_correction: bool = False) -> None:
    log.info('Assemble and correct NIfTI series')
    series_data = load_dixon_series(ds.tmp_nifti_series.value)
    if not skip_swap_correction:
        swaps_detected = swap_detection(series_data)
    resample_series_y(series_data)
    resample_series_z(series_data)
    create_mask_by_series(series_data)
    blended_volumes = blend_series_by_channel(series_data)
    log.info('Saving blended volumes before processing')
    save_dixon_volumes(blended_volumes, processed=False)

    if not skip_swap_correction:
        if any_swaps(swaps_detected):
            log.info('Swaps detected, attempting to correct')
            series_data = fix_fat_water_swaps(series_data, swaps_detected)
        else:
            log.info('No swaps detected')

    if skip_bias_correction:
        if not skip_swap_correction:
            blended_volumes = blend_series_by_channel(series_data)
    else:
        log.info('Estimate and correct bias field for NIfTI series')
        bias_field_correction(series_data)
        blended_volumes = blend_series_by_channel(series_data)
        log.info('Estimate and correct bias field for blended volumes')
        bias_field_correction(blended_volumes)

    log.info('Saving blended volumes after processing')
    save_dixon_volumes(blended_volumes)


def copy_liver_and_pancreas_series() -> None:
    """
    Biobank ID      DICOM SeriesName
    ----------      ----------------
    20202           t1_vibe_fs_tra_bh_pancreas
    20203           gre_mullti_echo_10_TE_liver (sp?)
    20204           ShMOLLI_192i (liver)
                    ShMOLLI_192i_T1MAP (liver)
                    ShMOLLI_192i_FITPARAMS (liver)
    20206           fl2d1_23_echos_pancreas
                    fl2d1_20_echos_pancreas_modified
    20254           LMS IDEAL OPTIMISED LOW FLIP 6DYN
    20259           ShMOLLI_192i_pancreas
                    ShMOLLI_192i_T1MAP_pancreas
                    ShMOLLI_192i_FITPARAMS_pancreas
    20260           gre_mullti_echo_10_TE_pancreas (sp?)
    """

    if not ds.nifti.exists():
        os.makedirs(ds.nifti.value)
    log.info('Copying liver and pancreas series')
    for file_name in os.listdir(ds.tmp_nifti_series.value):
        file_name = ds.tmp_nifti_series.path(file_name)
        volume = NiftiWrapper.from_filename(file_name)
        series_description = volume.get_meta('SeriesDescription').lower()
        image_type = volume.get_meta('ImageType')
        series_number = volume.get_meta('SeriesNumber')

        if series_description == 't1_vibe_fs_tra_bh_pancreas':
            # Dedicated pancreas volume series
            if 'NORM' in image_type:
                shutil.copyfile(file_name, ds.nifti.path(fn.pancreas_norm.value))
            else:
                shutil.copyfile(file_name, ds.nifti.path(fn.pancreas.value))
        elif series_description == 'gre_mullti_echo_10_te_liver':
            # Liver GRE
            if 'P' in image_type:
                shutil.copyfile(file_name, ds.nifti.path(fn.liver_phase.value))
            else:
                shutil.copyfile(file_name, ds.nifti.path(fn.liver_mag.value))
        elif series_description.startswith('lms') and series_description.find('ideal') > 0:
            # Liver IDEAL
            if 'P' in image_type:
                outfile = 'ideal_liver_phase_{}'.format(series_number)
            else:
                outfile = 'ideal_liver_magnitude_{}'.format(series_number)
            shutil.copyfile(file_name, ds.nifti.path('{}.{}'.format(outfile, NIFTI_EXT)))
        elif series_description == 'gre_mullti_echo_10_te_pancreas':
            # Pancreas GRE
            if 'P' in image_type:
                shutil.copyfile(file_name, ds.nifti.path(fn.pancreas_phase.value))
            else:
                shutil.copyfile(file_name, ds.nifti.path(fn.pancreas_mag.value))
        elif series_description == 'shmolli_192i':
            # Perspectum T1 mapping liver
            if 'P' in image_type:
                shutil.copyfile(file_name, ds.nifti.path(fn.shmolli_phase.value))
            else:
                shutil.copyfile(file_name, ds.nifti.path(fn.shmolli_mag.value))
        elif series_description == 'shmolli_192i_t1map':
            # Perspectum T1 mapping liver
            shutil.copyfile(file_name, ds.nifti.path(fn.shmolli_t1map.value))
        elif series_description == 'shmolli_192i_fitparams':
            # Perspectum T1 mapping liver
            shutil.copyfile(file_name, ds.nifti.path(fn.shmolli_params.value))

    list_of_files = os.listdir(ds.nifti.value)
    if len([x for x in list_of_files if 'ideal' in x]) > 0:
        log.info('Removing duplicate IDEAL series')
        for ideal_output in ['mag', 'phase']:
            files = [x for x in list_of_files if 'ideal' in x and ideal_output in x]
            os.rename(ds.nifti.path(sorted(files)[0]),
                      ds.nifti.path('ideal_liver_{}.{}'.format('magnitude' if ideal_output == 'mag' else 'phase',
                                                               NIFTI_EXT)))
            for file in sorted(files)[1:]:
                os.remove(ds.nifti.path(file))
    else:
        log.warning('No IDEAL series found')
