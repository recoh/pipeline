from pathlib import Path
from typing import List, Tuple, Union, Any

import SimpleITK as sitk
import nibabel as nib
import numpy as np
from scipy import ndimage as nd

from pipeline.constants import DataChannel as dc
from pipeline.logger import log


def get_project_root() -> Path:

    return Path(__file__).parent.parent


def channel_from_image_type(image_type: List[str]) -> str:
    if 'M' in image_type:
        return 'mag'
    if 'P' in image_type:
        return 'phase'

    return 'unknown'


def channel_to_str(data_channel: dc) -> str:
    channel_lookup = {
        dc.in_phase: 'ip',
        dc.opp_phase: 'op',
        dc.water: 'water',
        dc.fat: 'fat',
        dc.mask: 'mask',
        dc.midpoint: 'midpoint',
    }

    return '{}'.format(channel_lookup[data_channel])


def sitk_get_affine(vol: sitk.Image) -> np.array:
    origin = np.array(vol.GetOrigin())
    spacing = vol.GetSpacing()
    direction = np.reshape(vol.GetDirection(), (3, 3))
    affine = np.eye(4, 4, dtype='float64')
    affine[:3, :3] = spacing * direction
    affine[:3, 3] = origin

    return affine


def sitk_to_nibabel(vol: sitk.Image) -> nib.Nifti1Image:
    vol_np = sitk.GetArrayFromImage(vol)
    vol_np = np.transpose(vol_np)
    affine = sitk_get_affine(vol)
    affine[:2] *= -1

    return nib.Nifti1Image(vol_np, affine)


def nibabel_to_sitk(vol: nib.Nifti1Image) -> sitk.Image:
    data = np.transpose(np.asanyarray(vol.dataobj))
    header = vol.header
    spacing = np.array(header.get_zooms(), dtype='float64')
    affine = header.get_best_affine()
    affine[:2] *= -1
    direction = affine[:3, :3] / spacing
    origin = affine[:3, 3]
    vol_sitk = sitk.GetImageFromArray(data)
    vol_sitk.SetSpacing(spacing)
    vol_sitk.SetOrigin(origin)
    vol_sitk.SetDirection(direction.flatten())

    return vol_sitk


def convert_mm_to_voxels(mm: Tuple, spacing: Tuple) -> Tuple:
    if len(mm) == 2 and len(spacing) == 2:
        return round(mm[0] / spacing[0]), round(mm[1] / spacing[1])
    if len(mm) == 3 and len(spacing) == 3:
        return round(mm[0] / spacing[0]), round(mm[1] / spacing[1]), round(mm[2] / spacing[2])

    raise RuntimeError('Invalid input parameters')


def largest_connected_components(vol: np.ndarray, n_components: int = 1, structure: np.ndarray = None) -> np.ndarray:
    labels, n_features = nd.label(vol > 0, structure=structure)
    vol_label = np.zeros(vol.shape, dtype='int')
    if n_components > n_features:
        log.warning('Requested # of components = {}, image contains only {}'.format(n_components, n_features))
        n_components = n_features
    indices = np.argsort(nd.histogram(labels, 1, n_features, n_features))[::-1][:n_components] + 1
    for index, value in enumerate(indices, start=1):
        vol_label[labels == value] = index

    return vol_label


def sitk_largest_connected_components(vol: sitk.Image, n_components: int = 1,
                                      structure: np.ndarray = None) -> sitk.Image:
    vol_label_np = largest_connected_components(sitk.GetArrayFromImage(vol), n_components, structure)
    vol_label_sitk = sitk.GetImageFromArray(vol_label_np.astype('uint8'))
    vol_label_sitk.CopyInformation(vol)

    return vol_label_sitk


def bbox(data: np.ndarray, crop: bool = False, lcc: bool = False) -> Any:
    if lcc:
        mask = largest_connected_components(data)
    else:
        mask = data > 0
    pts = np.argwhere(mask)
    if len(pts) == 0:
        raise RuntimeError('Empty mask in call to bbox()')
    if len(data.shape) == 2:
        y_min, x_min = pts.min(axis=0)
        y_max, x_max = pts.max(axis=0)
        if not crop:
            return x_min, y_min, x_max, y_max
        else:
            return data[y_min:y_max + 1, x_min:x_max + 1]
    elif len(data.shape) == 3:
        z_min, y_min, x_min = pts.min(axis=0)
        z_max, y_max, x_max = pts.max(axis=0)
        if not crop:
            return x_min, y_min, z_min, x_max, y_max, z_max
        else:
            return data[z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
    elif len(data.shape) == 4:
        t_min, z_min, y_min, x_min = pts.min(axis=0)
        t_max, z_max, y_max, x_max = pts.max(axis=0)
        if not crop:
            return x_min, y_min, z_min, t_min, x_max, y_max, z_max, t_max
        else:
            return data[t_min:t_max + 1, z_min:z_max + 1, y_min:y_max + 1, x_min:x_max + 1]
    else:
        raise RuntimeError('Invalid data shape in call to bbox(): {}'.format(data.shape))


def sitk_bbox(vol: sitk.Image, crop: bool = False) -> Union[Tuple, sitk.Image]:
    x_min, y_min, z_min, x_max, y_max, z_max = bbox(sitk.GetArrayFromImage(vol))
    if not crop:
        return x_min, y_min, z_min, x_max, y_max, z_max

    return sitk.RegionOfInterest(vol,
                                 np.array([x_max - x_min + 1,
                                           y_max - y_min + 1,
                                           z_max - z_min + 1]).astype('uint32').tolist(),
                                 np.array([x_min, y_min, z_min]).astype('uint32').tolist())


def sitk_extract_slice(vol: sitk.Image, seg: sitk.Image, img: sitk.Image) -> (sitk.Image, sitk.Image):
    if len(img.GetSize()) == 4:
        img = sitk.Extract(img, img.GetSize()[:3] + (0,), (0, 0, 0, 0))

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(img.GetSpacing())
    resample.SetSize(img.GetSize())
    img_vol = resample.Execute(vol)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    img_seg = resample.Execute(seg)

    return img_vol, img_seg


def normalize(img: np.ndarray, threshold1: float = 99, threshold2: float = 0.975) -> np.ndarray:
    img = img.astype('float32') / np.percentile(img, threshold1)
    img[img > 1.0] = threshold2
    return img
