import os
from typing import Optional, List, Any, Tuple

import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.ndimage as nd
from skimage import segmentation
from PIL import Image

from pipeline.constants import DirectoryStructure as ds, FileNames as fn, PLOT_EXT
from pipeline.logger import log
from pipeline.util import sitk_bbox

ORTHOGONAL_PLANES = ['axial', 'coronal', 'sagittal']


def rescale(data: np.ndarray, lower_threshold: float = 0, upper_threshold: float = 255) -> np.ndarray:
    data = data.astype('float32')
    data -= np.min(data)
    m = np.max(data)
    if m != 0:
        data /= m

    return data * (upper_threshold - lower_threshold) + lower_threshold


def coronal_view(volume: nib.Nifti1Image, file_name: Optional[str] = None) -> Any:
    data = np.asanyarray(volume.dataobj)
    img = np.rot90(data[:, data.shape[1] // 2, :])
    img = rescale(img)
    img = img.astype('uint8')
    zooms = volume.header.get_zooms()
    img = Image.fromarray(img).resize((int(zooms[0] * img.shape[1]), int(zooms[2] * img.shape[0])))

    if file_name is not None:
        img.save(file_name)
    else:
        return img


def tile_multiple_slices(volume: sitk.Image, slice_range: List[int], plane: str = 'axial',
                         spacing: Tuple[float, float] = (1.0, 1.0), interpolator=sitk.sitkNearestNeighbor,
                         nrows: int = None, ncols: int = None) -> sitk.Image:
    if plane == ORTHOGONAL_PLANES[0]:
        img_slices = [volume[:, ::-1, z] for z in slice_range]
    elif plane == ORTHOGONAL_PLANES[1]:
        img_slices = [volume[:, y, ::-1] for y in slice_range]
    elif plane == ORTHOGONAL_PLANES[2]:
        img_slices = [volume[x, :, ::-1] for x in slice_range]
    else:
        raise ValueError('Must use {} planes'.format(ORTHOGONAL_PLANES))

    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(len(img_slices))))
        nrows = int(np.ceil(len(img_slices) / ncols))
    if nrows is None:
        nrows = int(np.ceil(len(img_slices) / ncols))
    if ncols is None:
        ncols = int(np.ceil(len(img_slices) / nrows))

    img_tiles = sitk.Tile(img_slices, [ncols, nrows])
    current_spacing = np.asanyarray(img_tiles.GetSpacing(), dtype='float32')
    current_size = np.asanyarray(img_tiles.GetSize(), dtype='float32')
    new_size = np.maximum((1, 1), np.round(current_size * current_spacing / np.asanyarray(spacing))).astype('uint16')
    new_image = sitk.Image(new_size.tolist(), sitk.sitkFloat32)
    new_image.SetOrigin(img_tiles.GetOrigin())
    new_image.SetDirection(img_tiles.GetDirection())
    new_image.SetSpacing(spacing)

    return sitk.Resample(img_tiles, new_image, sitk.Euler2DTransform(), interpolator)


def plot_body_data(opacity: float = 0.25) -> None:
    mask = sitk.ReadImage(ds.analysis.path(fn.body_mask.value))
    ip = sitk.Mask(sitk.RescaleIntensity(sitk.ReadImage(ds.nifti.path(fn.ip.value))), mask)
    fat_percent = sitk.Mask(sitk.RescaleIntensity(sitk.ReadImage(ds.analysis.path(fn.fat_percent.value))), mask)
    water_percent = sitk.Mask(sitk.RescaleIntensity(sitk.ReadImage(ds.analysis.path(fn.water_percent.value))), mask)

    if not ds.plots_analysis.exists():
        os.makedirs(ds.plots_analysis.value)

    for plane in ORTHOGONAL_PLANES:
        x_min, y_min, z_min, x_max, y_max, z_max = sitk_bbox(mask)
        if plane == ORTHOGONAL_PLANES[0]:
            top, bottom = z_max, z_min
        elif plane == ORTHOGONAL_PLANES[1]:
            top, bottom = y_max, y_min
        else:
            top, bottom = x_max, x_min
        if plane == ORTHOGONAL_PLANES[2]:
            slice_range = list(range(top - 16, bottom + 16, -8))
        else:
            slice_range = list(range(top - 24, bottom + 8, -8))
        if plane == ORTHOGONAL_PLANES[0]:
            nrows = None
        else:
            nrows = 2
        log.debug('In-phase: {}'.format(plane))
        img_tiles = tile_multiple_slices(sitk.Cast(ip, sitk.sitkUInt8), slice_range, plane, nrows=nrows)
        sitk.WriteImage(img_tiles, ds.plots_analysis.path('{}_{}.{}'.format('ip', plane, PLOT_EXT)))
        log.debug('Body mask: {}'.format(plane))
        seg_tiles = tile_multiple_slices(mask, slice_range, plane, nrows=nrows)
        label_overlay = sitk.LabelOverlay(image=img_tiles, labelImage=seg_tiles, opacity=opacity)
        sitk.WriteImage(label_overlay, ds.plots_analysis.path('{}_{}.{}'.format('mask', plane, PLOT_EXT)))
        log.debug('Fat percentages: {}'.format(plane))
        seg_tiles = tile_multiple_slices(sitk.Cast(fat_percent, sitk.sitkUInt8), slice_range, plane, nrows=nrows)
        sitk.WriteImage(seg_tiles, ds.plots_analysis.path('{}_{}.{}'.format('fat_percent', plane, PLOT_EXT)))
        log.debug('Water percentages: {}'.format(plane))
        seg_tiles = tile_multiple_slices(sitk.Cast(water_percent, sitk.sitkUInt8), slice_range, plane, nrows=nrows)
        sitk.WriteImage(seg_tiles, ds.plots_analysis.path('{}_{}.{}'.format('water_percent', plane, PLOT_EXT)))


def nparray_show(arr: np.ndarray, lut: str = 'nipy_spectral', title: str = None, filename: str = None,
                 colorbar: bool = True) -> None:
    fig = plt.figure(figsize=(12, 6.4))
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    plt.imshow(arr[::-1], interpolation='nearest')
    ax.set_aspect('equal')
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.set_cmap(lut)
    if colorbar:
        plt.colorbar(orientation='vertical')
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        log.error('Cannot display to terminal')
    plt.close(fig)


def landmarks_as_spheres(data: np.ndarray, radius: int = 8,
                         spacing: Tuple[float, float, float] = (1, 1, 1)) -> np.ndarray:
    small_spheres = nd.distance_transform_edt(data == 0, sampling=spacing) < radius
    if len(np.unique(data)) > 2:
        small_spheres = segmentation.watershed(small_spheres, data, mask=small_spheres)

    return small_spheres
