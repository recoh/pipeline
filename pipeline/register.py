import json
import os
import random
import re
import time
from functools import partial
from multiprocessing import cpu_count, get_context

import SimpleITK as sitk

from pipeline.constants import DirectoryStructure as ds, FileNames as fn
from pipeline.logger import log


def multires_registration(fixed_image: sitk.Image, moving_image: sitk.Image,
                          initial_transform: sitk.Transform, verbose: bool = False) -> (sitk.Transform, float):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      estimateLearningRate=registration_method.Once)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(fixed_image, moving_image)
    if verbose:
        log.debug(f'Final metric value: {registration_method.GetMetricValue():.4f}')
        log.debug(f"Optimizer's stopping condition, {registration_method.GetOptimizerStopConditionDescription()}")

    return final_transform, registration_method.GetMetricValue()


def warp_landmarks(subject: str, reference_directory: str, fixed_img_name: str, moving_img_name: str) -> None:
    log.debug(f'Warping {moving_img_name} on {subject}')
    fixed_image = sitk.ReadImage(fixed_img_name, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(os.path.join(reference_directory, moving_img_name, fn.body_mask.value),
                                  sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image,
                                                          sitk.AffineTransform(fixed_image.GetDimension()),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    final_transform, _ = multires_registration(fixed_image, moving_image, initial_transform)

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0,
                                     sitk.sitkUInt8)
    landmarks_image = sitk.ReadImage(os.path.join(reference_directory, moving_img_name, fn.bone_joints.value),
                                     sitk.sitkUInt8)
    landmarks_resampled = sitk.Resample(landmarks_image, fixed_image, final_transform, sitk.sitkNearestNeighbor, 0.0,
                                        sitk.sitkUInt8)

    output_folder = os.path.join(ds.tmp.value, 'warp_landmarks', f'{moving_img_name}_to_{subject}')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    sitk.WriteImage(moving_resampled, os.path.join(output_folder, 'moving_resampled.nii.gz'))
    sitk.WriteImage(landmarks_resampled, os.path.join(output_folder, 'landmarks_resampled.nii.gz'))
    sitk.WriteTransform(final_transform, os.path.join(output_folder, 'moving_resampled.tfm'))


def atlas_landmarks(reference_directory: str, n_images: int = 35) -> None:
    if ds.analysis.exists(fn.bone_joints_initial_mask.value):
        log.warning('Atlas-based registration already exists for bone joints!')
        return

    reader = sitk.ImageFileReader()
    reader.SetFileName(ds.nifti.path(fn.ip.value))
    reader.ReadImageInformation()
    dicom_id = json.loads(reader.GetMetaData('descrip'))['dicom']

    reference_images = list()
    for root, dirs, _ in os.walk(os.path.join(reference_directory)):
        for dir in dirs:
            ref_img = os.path.join(root, dir, fn.bone_joints.value)
            if os.path.exists(ref_img):
                reference_images.append(ref_img)
    reference_scans = [re.match(r".*/([0-9][0-9][0-9][0-9])/", x).group(1) for x in reference_images]
    log.info(f'{len(reference_scans)} reference subjects found for the bone-joints')
    reference_scans = random.sample(reference_scans, min(n_images, len(reference_scans)))
    log.info(f'{len(reference_scans)} reference subjects used for the bone-joints')
    if n_images < 1:
        log.error('Cannot perform atlas-based registration, please check spelling')
        return

    fixed_img_name = ds.analysis.path(fn.body_mask.value)
    start_time = time.time()

    # Serial versus parallel
    # for moving_subject in reference_scans:
    #     warp_landmarks(dicom_id, reference_directory, fixed_img_name, moving_subject)

    cpu_used = cpu_count() * 2 // 3
    chunk_size = round(n_images / cpu_used)
    log.debug(f'{cpu_used} CPUs allocated and chunksize = {chunk_size:d}')
    warp_landmarks_partial = partial(warp_landmarks, dicom_id, reference_directory, fixed_img_name)
    with get_context('spawn').Pool(processes=cpu_used) as pool:  # with Pool(processes=cpu_used) as pool:
        pool.map(warp_landmarks_partial, reference_scans, chunksize=int(chunk_size))
        pool.close()
    elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
    log.info(f'Atlas-based registration for {n_images} subjects completed in {elapsed_time}')

    fixed_img = sitk.ReadImage(fixed_img_name)
    size = fixed_img.GetSize()
    labels = sitk.Image(size, sitk.sitkUInt8)
    labels.CopyInformation(fixed_img)
    n_failures = 0
    for moving_subject in reference_scans:
        warp_seg = os.path.join(ds.tmp.value, 'warp_landmarks', f'{moving_subject}_to_{dicom_id}',
                                'landmarks_resampled.nii.gz')
        if os.path.exists(warp_seg):
            labels += sitk.MaskNegated(sitk.ReadImage(warp_seg, sitk.sitkUInt8), labels)
        else:
            n_failures += 1
            n_images -= 1
    log.warning(f'{n_failures} registration failures') if n_failures > 0 else None
    sitk.WriteImage(labels, ds.analysis.path(fn.bone_joints_initial_mask.value))
