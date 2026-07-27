from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from loguru import logger

from .config import SegmentationProcessing


@dataclass
class NamedLabelImage:
    lut: dict[str, int]
    image: sitk.Image


def create_canvas_for_volumes(volumes: list[sitk.Image], spacing: tuple[float, ...]) -> sitk.Image:
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


def make_contiguous(label1: sitk.Image, label2: sitk.Image, closing_radius: tuple[int, int, int]) -> tuple[sitk.Image, sitk.Image]:
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

        if options.close:
            roi = sitk.GrayscaleMorphologicalClosing(roi, options.closing_radius)
        roi = sitk.BinaryFillhole(roi)
        if options.median_filter:
            roi = sitk.BinaryMedian(roi, options.median_filter_radius)
        if options.open:
            roi = sitk.GrayscaleMorphologicalOpening(roi, options.opening_radius)
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
        logger.info(f"Adding label {label}")
        processed_image += image * label
        processed_image[processed_image > label] = label

    processed_labelmap = NamedLabelImage(image=processed_image, lut=segmentation.lut)

    return processed_labelmap
