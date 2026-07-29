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
    disk.read_images(data_files, filenames=["tibia.nrrd", "tibia_cartilage.nrrd", "femur.nrrd", "femur_cartilage.nrrd"])


def test_read_bad_image(empty_file):
    with pytest.raises(ValueError):
        disk.read_image(empty_file)


def test_stl_roundtrip(simple_meshes, tmp_path):
    name, mesh = next(iter(simple_meshes.items()))
    disk.write_stl(mesh, tmp_path / f"{name}.stl")
    new_mesh = disk.read_stl(tmp_path / f"{name}.stl")
    assert new_mesh.GetNumberOfPoints() == mesh.GetNumberOfPoints()


def test_ply_roundtrip(simple_meshes, tmp_path):
    name, mesh = next(iter(simple_meshes.items()))
    disk.write_ply(mesh, tmp_path / f"{name}.ply")
    new_mesh = disk.read_ply(tmp_path / f"{name}.ply")
    assert new_mesh.GetNumberOfPoints() == mesh.GetNumberOfPoints()


def test_vtp_roundtrip(simple_meshes, tmp_path):
    name, mesh = next(iter(simple_meshes.items()))
    disk.write_vtp(mesh, tmp_path / f"{name}.vtp")
    new_mesh = disk.read_vtp(tmp_path / f"{name}.vtp")
    assert new_mesh.GetNumberOfPoints() == mesh.GetNumberOfPoints()


def test_obj_roundtrip(simple_meshes, tmp_path):
    name, mesh = next(iter(simple_meshes.items()))
    disk.write_obj(mesh, tmp_path / f"{name}.obj")
    new_mesh = disk.read_obj(tmp_path / f"{name}.obj")
    assert new_mesh.GetNumberOfPoints() == mesh.GetNumberOfPoints()
