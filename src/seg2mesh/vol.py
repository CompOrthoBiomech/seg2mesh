from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from loguru import logger
from SimpleITK.utilities.vtk import sitk2vtk
from vtkmodules.all import vtkImageData

from .config import SegmentationProcessing

IDENTITY_TRANSFORM = sitk.Transform()


@dataclass
class NamedLabelImage:
    lut: dict[str, int]
    image: sitk.Image


@dataclass
class NamedVTKImage:
    lut: dict[str, int]
    image: vtkImageData


def create_canvas_for_volumes(volumes: list[sitk.Image], spacing: tuple[float, ...]) -> sitk.Image:
    """
    Create a unified canvas image from a list of volumes and target spacing.

    :param volumes: List of SimpleITK Image objects to resample.
    :param spacing: The target spacing for the canvas.

    :return: A SimpleITK Image representing the unified canvas.
    """
    origin = np.zeros((len(volumes), 3))
    corner = np.zeros((len(volumes), 3))
    for i, vol in enumerate(volumes):
        start_point = vol.TransformIndexToPhysicalPoint([0, 0, 0])
        end_point = vol.TransformIndexToPhysicalPoint([s - 1 for s in vol.GetSize()])
        origin[i, :] = np.min([start_point, end_point], axis=0)
        corner[i, :] = np.max([start_point, end_point], axis=0)

    origin = np.min(origin, axis=0)
    corner = np.max(corner, axis=0)
    physical_size = corner - origin
    img_size = [int(np.ceil(physical_size[i] / spacing[i])) for i in range(3)]
    canvas = sitk.Image(*img_size, sitk.sitkUInt8)
    canvas.SetOrigin(origin)
    canvas.SetSpacing(spacing)
    return canvas


def resample_volumes_to_canvas(volumes: list[sitk.Image], canvas: sitk.Image) -> NamedLabelImage:
    """
    Resample image volumes to a unified canvas. `create_canvas_for_volumes()` can be used to create a suitable canvas.

    :param volumes: List of SimpleITK Image objects to resample.
    :param canvas: The target canvas to resample to.

    :return: A NamedLabelImage with the resampled volumes and LUT mapping label names to integer values.
    """
    composite = NamedLabelImage(image=canvas, lut={})
    lut = {}
    for i, v in enumerate(volumes):
        try:
            lut[v["Short Name"]] = i + 1
        except KeyError:
            logger.warning(
                f"Short Name not found for volume {i + 1}, LUT key will just be integer label. Recommend adding 'Short Name' to image metadata."
            )
            lut[str(i + 1)] = i + 1

        resampled = sitk.Resample(
            sitk.Cast(v, sitk.sitkUInt8),
            canvas,
            transform=sitk.Transform(),
            interpolator=sitk.sitkLabelLinear,
        )
        composite.image += resampled * (i + 1)
        composite.image[composite.image > (i + 1)] = i + 1
    composite.lut = lut
    return composite


def resample_label_image(
    image: sitk.Image, spacing: tuple[float, float, float], transform: sitk.Transform = IDENTITY_TRANSFORM
) -> sitk.Image:
    """
    Resamples a label image to the specified spacing and SimpleITK Transform (default is identity).
    The sitkLabelLinear interpolator is used to better preserve labels during resampling.

    :param image: The input label image to resample.
    :param spacing: The desired spacing for the output image.
    :param transform: The transform to apply during resampling.

    :returns: The resampled label image.
    """
    output_size = [int(np.ceil(orig_s / s * n)) for s, orig_s, n in zip(spacing, image.GetSpacing(), image.GetSize())]
    return sitk.Resample(
        image,
        output_size,
        transform=transform,
        outputOrigin=image.GetOrigin(),
        outputSpacing=spacing,
        outputDirection=image.GetDirection(),
        interpolator=sitk.sitkLabelLinear,
        defaultPixelValue=0,
    )


def resample_greyscale(image: sitk.Image, spacing: tuple[float, float, float] | None = None) -> sitk.Image:
    if spacing is None:
        spacing = image.GetSpacing()

    start_point = image.TransformIndexToPhysicalPoint([0, 0, 0])
    end_point = image.TransformIndexToPhysicalPoint([s - 1 for s in image.GetSize()])
    origin = np.min([start_point, end_point], axis=0)
    corner = np.max([start_point, end_point], axis=0)
    physical_size = corner - origin
    img_size = [int(np.ceil(physical_size[i] / spacing[i])) for i in range(3)]  # type: ignore
    canvas = sitk.Image(*img_size, sitk.sitkUInt8)
    canvas.SetOrigin(origin)
    canvas.SetSpacing(spacing)
    resampled = sitk.Resample(
        image,
        canvas,
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
    )
    return resampled


def create_lut_from_label_image(label_image: sitk.Image) -> dict[str, int]:
    labels = sitk.LabelShapeStatisticsImageFilter()
    labels.Execute(label_image)
    return {str(label): int(label) for label in labels.GetLabels()}


def remove_islands(image: sitk.Image) -> sitk.Image:
    """
    Remove islands (by keeping only the largest connected component) in a binary image.

    :param image: The binary image to process.
    :return: The image with islands removed.
    """
    components = sitk.ConnectedComponentImageFilter()
    components.SetFullyConnected(False)
    connected = components.Execute(image)
    sorted_labels = sitk.RelabelComponent(connected, sortByObjectSize=True)
    return sorted_labels == 1


def make_contiguous(label1: sitk.Image, label2: sitk.Image, closing_radius: tuple[int, int, int]) -> tuple[sitk.Image, sitk.Image]:
    """
    Adjust `label1` and `label2` to be contiguous. `label1` takes precedence over `label2`

    :param label1: The first label image.
    :param label2: The second label image.
    :param closing_radius: The radius (in voxels) for dilation and morphological closing operations.

    :return: A tuple of the adjusted `label1` and `label2` images.
    """
    labelstats = sitk.LabelShapeStatisticsImageFilter()
    labelstats.Execute(label1)
    bbox = labelstats.GetBoundingBox(1)
    pad = [c * 2 for c in closing_radius]
    roi1 = sitk.ConstantPad(sitk.RegionOfInterest(label1, bbox[3:6], bbox[0:3]), padUpperBound=pad, padLowerBound=pad, constant=0)
    roi2 = sitk.ConstantPad(sitk.RegionOfInterest(label2, bbox[3:6], bbox[0:3]), padUpperBound=pad, padLowerBound=pad, constant=0)

    union = sitk.Or(roi1, roi2)
    union = sitk.GrayscaleMorphologicalClosing(union, closing_radius)
    dilate_label1 = sitk.BinaryDilate(roi1, closing_radius)
    dilate_label2 = sitk.BinaryDilate(roi2, closing_radius)
    filled_label1 = sitk.Or(roi1, sitk.And(dilate_label1, dilate_label2))
    filled_label1 *= union
    filled_label1 = sitk.BinaryMedian(filled_label1, closing_radius) * sitk.Not(roi2)
    filled_label1 = sitk.Resample(filled_label1, interpolator=sitk.sitkLabelLinear, referenceImage=label1)
    adjusted_label2 = sitk.And(label2, sitk.Not(filled_label1))

    return filled_label1, adjusted_label2


def process_segmentation(segmentation: NamedLabelImage, options: SegmentationProcessing) -> NamedLabelImage:
    """
    Apply a processing workflow to the segmentation.

    :param segmentation: The input segmentation as a NamedLabelImage.
    :param options: The processing options as a SegmentationProcessing object.

    :return: The processed segmentation as a NamedLabelImage.
    """
    processed_image = sitk.Image(segmentation.image.GetSize(), segmentation.image.GetPixelID())
    processed_image.CopyInformation(segmentation.image)

    processed_labels = {}
    labelstats = sitk.LabelShapeStatisticsImageFilter()
    labelstats.Execute(segmentation.image)
    for name, label in segmentation.lut.items():
        logger.info(f"Processing {name} with integer label {label}")
        bbox = labelstats.GetBoundingBox(label)
        roi = sitk.ConstantPad(
            sitk.RegionOfInterest(segmentation.image, bbox[3:6], bbox[0:3]),
            padUpperBound=[c * 2 for c in options.closing_radius],
            padLowerBound=[c * 2 for c in options.closing_radius],
            constant=0,
        )
        roi = sitk.BinaryThreshold(roi, lowerThreshold=label, upperThreshold=label, insideValue=1, outsideValue=0)
        roi = remove_islands(roi)
        if options.spur_removal_length > 0:
            roi = sitk.BinaryPruning(roi, iteration=options.spur_removal_length)

        if options.close:
            roi = sitk.GrayscaleMorphologicalClosing(roi, options.closing_radius)

        if name in options.open_list:
            roi = sitk.GrayscaleMorphologicalOpening(roi, options.opening_radius)

        roi = sitk.BinaryFillhole(roi)
        roi = sitk.Resample(roi, processed_image, interpolator=sitk.sitkLabelLinear)
        processed_labels[label] = roi
    for name1, name2 in options.make_contiguous:
        try:
            logger.info(f"Making {name1} and {name2} contiguous")
            processed_labels[segmentation.lut[name1]], processed_labels[segmentation.lut[name2]] = make_contiguous(
                processed_labels[segmentation.lut[name1]],
                processed_labels[segmentation.lut[name2]],
                closing_radius=options.contiguous_closing_radius,
            )
        except KeyError as e:
            logger.error(f"LUT is missing key: {e}")
    for label, image in sorted(processed_labels.items()):
        processed_image += image * label
        processed_image[processed_image > label] = label

    processed_labelmap = NamedLabelImage(image=processed_image, lut=segmentation.lut)

    return processed_labelmap


def convert_segmentation_to_vtk(segmentation: NamedLabelImage) -> NamedVTKImage:
    """
    Converts a NamedLabelImage  to a NamedVTKImage.

    :param segmentation: The NamedLabelImage image to convert.

    :returns: A NamedVTKImage with converted image data and same lookup table.
    """
    vtk_image = sitk2vtk(segmentation.image)
    vtk_image.GetPointData().GetScalars().SetName("Label")
    return NamedVTKImage(image=vtk_image, lut=segmentation.lut)
