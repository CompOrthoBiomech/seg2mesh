import numpy as np
import SimpleITK as sitk

from .config import SegmentationProcessing


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


def resample_volumes_to_canvas(volumes: list[sitk.Image], canvas: sitk.Image) -> sitk.Image:
    composite = sitk.Image(canvas)
    for i, v in enumerate(volumes):
        resampled = sitk.Resample(
            sitk.Cast(v, sitk.sitkUInt8),
            canvas,
            transform=sitk.Transform(),
            interpolator=sitk.sitkNearestNeighbor,
        )
        composite += resampled * (i + 1)
        composite[composite > (i + 1)] = i + 1
    return composite


def process_segmentation(segmentation: sitk.Image, options: SegmentationProcessing) -> sitk.Image:
    processed_image = sitk.Image(segmentation)

    processed_labels = []
    labelstats = sitk.LabelShapeStatisticsImageFilter()
    labelstats.Execute(segmentation)
    for label in labelstats.GetLabels():
        bbox = labelstats.GetBoundingBox(label)
        roi = sitk.ConstantPad(
            sitk.RegionOfInterest(segmentation == label, bbox),
            padUpperBound=options.closing_radius,
            padLowerBound=options.closing_radius,
            constant=0,
        )
        if options.close:
            roi = sitk.GrayscaleMorphologicalClosing(roi, options.closing_radius)
        roi = sitk.BinaryFillhole(roi)
        if options.median_filter:
            roi = sitk.BinaryMedian(roi, options.median_filter_radius)
        if options.open:
            roi = sitk.GrayscaleMorphologicalOpening(roi, options.opening_radius)
        roi = sitk.Resample(roi, interpolator=sitk.sitkNearestNeighbor, referenceImage=processed_image) * label
        processed_labels.append(roi)
    # for label1, label2 in options.make_contiguous:
    #     processed_labels[label1 - 1] = make_contiguous(processed_labels[label1 - 1], processed_labels[label2 - 1])
    for label in processed_labels:
        processed_image += label

    return processed_image


def make_contiguous(label1: sitk.Image, label2: sitk.Image) -> sitk.Image:
    union = sitk.Or(label1, label2)
    union = sitk.GrayscaleMorphologicalClosing(union, [3, 3, 3])
    dilate_label1 = sitk.BinaryDilate(label1, [3, 3, 3])
    dilate_label2 = sitk.BinaryDilate(label2, [3, 3, 3])
    filled_label1 = sitk.Or(label1, sitk.And(dilate_label1, dilate_label2))
    filled_label1 *= union
    filled_label1 = sitk.BinaryMedian(filled_label1, [1, 1, 1]) * sitk.Not(label2)
    return filled_label1
