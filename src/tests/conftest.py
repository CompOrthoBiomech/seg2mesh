import logging

import pytest
import SimpleITK as sitk
from loguru import logger


class PropagateHandler(logging.Handler):
    def emit(self, record):
        logging.getLogger(record.name).handle(record)


@pytest.fixture(autouse=True)
def setup_loguru_pytest_integration():
    logger.remove()
    handler_id = logger.add(PropagateHandler(), format="{message}")
    yield
    logger.remove(handler_id)


@pytest.fixture(scope="session")
def simple_anatomies() -> dict[str, sitk.Image]:
    final = {}
    gaussian = sitk.GaussianSource(sitk.sitkUInt8, size=(50, 50, 50), mean=(25, 25, 25), sigma=(8, 8, 8))
    anatomy = []
    for translate in [(0, -15, 0), (0, -11, 0), (0, 15, 0), (0, 11, 0)]:
        tx = sitk.TranslationTransform(3, translate)
        translated_image = sitk.Resample(gaussian, gaussian, tx, sitk.sitkNearestNeighbor, 0.0, gaussian.GetPixelID())
        anatomy.append(sitk.BinaryThreshold(translated_image, lowerThreshold=150.0, upperThreshold=255.0, insideValue=1, outsideValue=0))
    anatomy[1] = (anatomy[1] & ~anatomy[0]) * 2
    anatomy[3] = (anatomy[3] & ~anatomy[2]) * 4
    anatomy[2] *= 3
    final["knee"] = sum(anatomy)
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(final["knee"])
    for label, name in enumerate(("tibia", "tibia_cartilage", "femur", "femur_cartilage")):
        bbox = shape_stats.GetBoundingBox(label + 1)
        final[name] = sitk.Extract(final["knee"] == label + 1, bbox[3::], bbox[0:3])
    return final


@pytest.fixture(scope="session")
def data_files(tmp_path_factory, simple_anatomies):
    file_dir = tmp_path_factory.mktemp("data")
    for ftype in (".nii", ".mhd", ".nrrd"):
        sitk.WriteImage(simple_anatomies["knee"], file_dir / f"knee{ftype}")
    for name, anatomy in simple_anatomies.items():
        if name == "knee":
            continue
        sitk.WriteImage(anatomy, file_dir / f"{name}.nii")
    return file_dir
