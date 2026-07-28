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


def read_images(dirpath: Path | str, filenames: list[str]) -> list[sitk.Image]:
    dirpath = Path(dirpath)
    images = []
    for file in filenames:
        images.append(read_image(dirpath.joinpath(file)))
    return images


def read_stl(filepath: Path | str) -> vtk.vtkPolyData:
    reader = vtk.vtkSTLReader()
    reader.SetFileName(Path(filepath).as_posix())
    logger.info(f"Reading STL file: {filepath}")
    reader.Update()
    return reader.GetOutput()


def read_ply(filepath: Path | str) -> vtk.vtkPolyData:
    reader = vtk.vtkPLYReader()
    reader.SetFileName(Path(filepath).as_posix())
    logger.info(f"Reading PLY file: {filepath}")
    reader.Update()
    return reader.GetOutput()


def read_obj(filepath: Path | str) -> vtk.vtkPolyData:
    reader = vtk.vtkOBJReader()
    reader.SetFileName(Path(filepath).as_posix())
    logger.info(f"Reading OBJ file: {filepath}")
    reader.Update()
    return reader.GetOutput()


def read_vtp(filepath: Path | str) -> vtk.vtkPolyData:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(Path(filepath).as_posix())
    logger.info(f"Reading VTP file: {filepath}")
    reader.Update()
    return reader.GetOutput()


def write_image(data: sitk.Image, filepath: Path | str) -> None:
    logger.info(f"Writing image to {filepath}")
    sitk.WriteImage(data, filepath)


def write_stl(data: vtk.vtkPolyData, filepath: Path | str) -> None:
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(Path(filepath).as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_ply(data: vtk.vtkPolyData, filepath: Path | str) -> None:
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(Path(filepath).as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_obj(data: vtk.vtkPolyData, filepath: Path | str) -> None:
    writer = vtk.vtkOBJWriter()
    writer.SetFileName(Path(filepath).as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_vtp(data: vtk.vtkPolyData, filepath: Path | str) -> None:
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(Path(filepath).as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_vtu(data: vtk.vtkUnstructuredGrid, filepath: Path | str) -> None:
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(Path(filepath).as_posix())
    writer.SetInputData(data)
    writer.Update()


def write_vti(data: vtk.vtkImageData, filepath: Path | str) -> None:
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(Path(filepath).as_posix())
    writer.SetInputData(data)
    writer.Update()
