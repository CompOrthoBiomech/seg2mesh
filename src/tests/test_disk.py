from pathlib import Path

import pytest

from seg2mesh import disk


@pytest.fixture
def empty_file(tmp_path) -> Path:
    file = tmp_path / "empty.nii"
    file.touch()
    return file


def test_image_roundtrip(data_files, tmp_path):
    for ftype in (".nii", ".mhd", ".nrrd"):
        image = disk.read_image(data_files / f"knee{ftype}")
        assert image is not None
        disk.write_image(image, tmp_path / f"knee{ftype}")


def test_read_images(data_files):
    disk.read_images(data_files, filenames=["tibia.nii", "tibia_cartilage.nii", "femur.nii", "femur_cartilage.nii"])


def test_read_bad_image(empty_file):
    with pytest.raises(ValueError):
        disk.read_image(empty_file)
