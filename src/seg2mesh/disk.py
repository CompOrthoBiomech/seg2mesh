import json
from pathlib import Path

import SimpleITK as sitk
import vtk
from loguru import logger


def read_image(filepath: Path | str) -> sitk.Image:
    reader = sitk.ImageFileReader()
    ftype = reader.GetImageIOFromFileName(filepath)
    filepath = Path(filepath)
    if ftype == "":
        message = f"Unsupported file type: {filepath}"
        logger.error(message)
        raise ValueError(message)
    logger.info(f"{filepath} detected as {ftype}, importing...")
    image = sitk.ReadImage(filepath)
    image["Short Name"] = filepath.stem
    return image


def read_images(filepaths: list[Path | str]) -> list[sitk.Image]:
    images = []
    for file in filepaths:
        images.append(read_image(Path(file)))
    return images


def read_stl(filepath: Path | str) -> vtk.vtkPolyData:
    reader = vtk.vtkSTLReader()
    reader.SetFileName(Path(filepath).as_posix())
    logger.info(f"Reading STL file from {filepath}")
    reader.Update()
    return reader.GetOutput()


def read_ply(filepath: Path | str) -> vtk.vtkPolyData:
    reader = vtk.vtkPLYReader()
    reader.SetFileName(Path(filepath).as_posix())
    logger.info(f"Reading PLY file from {filepath}")
    reader.Update()
    return reader.GetOutput()


def read_obj(filepath: Path | str) -> vtk.vtkPolyData:
    reader = vtk.vtkOBJReader()
    reader.SetFileName(Path(filepath).as_posix())
    logger.info(f"Reading OBJ file from {filepath}")
    reader.Update()
    return reader.GetOutput()


def read_vtp(filepath: Path | str) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(Path(filepath).as_posix())
    logger.info(f"Reading VTP file from {filepath}")
    reader.Update()
    return reader.GetOutput()


def write_image(data: sitk.Image, filepath: Path | str) -> None:
    logger.info(f"Writing image to {filepath}")
    sitk.WriteImage(data, filepath)


def read_lut(filepath: Path | str) -> dict[str, int]:
    filepath = Path(filepath)
    logger.info(f"Reading LUT from {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def write_lut(lut: dict[str, int], filepath: Path | str) -> None:
    logger.info(f"Writing LUT to {filepath}")
    with open(filepath, "w") as f:
        f.write(json.dumps(lut, indent=4))


def write_stl(data: vtk.vtkPolyData, filepath: Path | str) -> None:
    filepath = Path(filepath)
    logger.info(f"Writing STL file to {filepath}")
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(filepath.as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_ply(data: vtk.vtkPolyData, filepath: Path | str) -> None:
    filepath = Path(filepath)
    logger.info(f"Writing PLY file to {filepath}")
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(filepath.as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_obj(data: vtk.vtkPolyData, filepath: Path | str) -> None:
    filepath = Path(filepath)
    logger.info(f"Writing OBJ file to {filepath}")
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(filepath.as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_vtp(data: vtk.vtkPolyData, filepath: Path | str) -> None:
    filepath = Path(filepath)
    logger.info(f"Writing VTP file to {filepath}")
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filepath.as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_vtu(data: vtk.vtkUnstructuredGrid, filepath: Path | str) -> None:
    filepath = Path(filepath)
    logger.info(f"Writing VTU file to {filepath}")
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filepath.as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_vti(data: vtk.vtkImageData, filepath: Path | str) -> None:
    filepath = Path(filepath)
    logger.info(f"Writing VTI file to {filepath}")
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filepath.as_posix())
    writer.SetInputData(data)
    writer.Update()
