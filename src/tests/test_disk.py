from pathlib import Path

import pytest

from seg2mesh import disk


def test_read_image(data_files):
    for ftype in (".nii", ".mhd", ".nrrd"):
        image = disk.read_image(data_files / f"knee{ftype}")
        assert image is not None


def test_read_bad_image():
    with pytest.raises(ValueError):
        disk.read_image(Path(__file__).parent.joinpath("junk.nii"))
