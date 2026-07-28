import pytest

from seg2mesh import vol
from seg2mesh.config import SegmentationProcessing


@pytest.fixture
def segmentation(simple_anatomies):
    return vol.NamedLabelImage(image=simple_anatomies["knee"], lut={"tibia": 1, "tibia_cartilage": 2, "femur": 3, "femur_cartilage": 4})


def test_create_canvas_for_volumes(simple_anatomies):
    canvas = vol.create_canvas_for_volumes(
        [simple_anatomies[s] for s in ("tibia", "tibia_cartilage", "femur", "femur_cartilage")], spacing=(0.5, 0.5, 0.5)
    )
    resampled = vol.resample_volumes_to_canvas(
        volumes=[simple_anatomies[s] for s in ("tibia", "tibia_cartilage", "femur", "femur_cartilage")], canvas=canvas
    )
    assert canvas.GetSpacing() == pytest.approx(resampled.image.GetSpacing())
    assert canvas.GetSize() == resampled.image.GetSize()


def test_process_segmentation(segmentation):
    processed = vol.process_segmentation(segmentation, SegmentationProcessing())
    assert processed.image.GetSize() == segmentation.image.GetSize()
    vol.convert_segmentation_to_vtk(processed)
