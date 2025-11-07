import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pyacvd
import SimpleITK as sitk
import vtkmodules.all as vtk
from pyvista import PolyData
from vtkmodules.util import numpy_support


@dataclass
class Config:
    input_dir: str = "."
    voxel_resample_length: float = 0.3
    closing_radius: int = 3
    smoothing_distance: float = 0.3
    smoothing_relaxation_factor: float = 0.01
    smoothing_iterations: int = 1000
    remesh_edge_length: float = 1.0
    output_dir: str = "output"
    output_format: Literal["vtp", "stl"] = "vtp"


def main(config: Config):
    volumes = []
    volume_names = []
    for i, file in enumerate(Path(config.input_dir).glob("*.nii")):
        img = sitk.ReadImage(file.as_posix(), outputPixelType=sitk.sitkUInt8)
        scale = [s / config.voxel_resample_length for s in img.GetSpacing()]
        target_dim = [int(s * d + 0.5) for (s, d) in zip(scale, img.GetSize())]
        origin = img.GetOrigin()
        upsampled = sitk.Resample(
            img,
            target_dim,
            transform=sitk.Transform(),
            interpolator=sitk.sitkNearestNeighbor,
            outputOrigin=origin,
            outputSpacing=[config.voxel_resample_length] * 3,
            outputDirection=img.GetDirection(),
        )
        label = sitk.GrayscaleMorphologicalClosing(upsampled, [config.closing_radius] * 3) * (i + 1)
        volumes.append(label)
        volume_names.append(file.stem)
        print(f"Added resampled {file} to volumes")

    composite = volumes[0]
    for i, volume in enumerate(volumes[1:]):
        composite += volume
        composite[composite > (i + 2)] = i + 2
    composite = sitk.ConstantPad(composite, padLowerBound=(1, 1, 1), padUpperBound=(1, 1, 1), constant=0)

    nparray = sitk.GetArrayFromImage(composite)
    vtk_data = numpy_support.numpy_to_vtk(nparray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtkimage = vtk.vtkImageData()
    vtkimage.SetDimensions(nparray.shape[2], nparray.shape[1], nparray.shape[0])
    vtkimage.SetSpacing(composite.GetSpacing())
    vtkimage.SetOrigin(composite.GetOrigin())
    vtkimage.GetPointData().SetScalars(vtk_data)

    output_path = Path(config.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path.joinpath("config.json"), "w") as f:
        json.dump(asdict(config), fp=f, indent=4)

    for i, name in enumerate(volume_names):
        snets = vtk.vtkSurfaceNets3D()
        snets.SetInputData(vtkimage)
        snets.GenerateLabels(len(volumes), 1, len(volumes))
        snets.SetOutputStyleToSelected()
        snets.GetSmoother().SetNumberOfIterations(config.smoothing_iterations)
        snets.GetSmoother().SetConstraintDistance(config.smoothing_distance)
        snets.GetSmoother().SetRelaxationFactor(config.smoothing_relaxation_factor)
        snets.AddSelectedLabel(i + 1)
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(snets.GetOutputPort())
        clean.PointMergingOff()
        clean.ConvertLinesToPointsOff()
        clean.ConvertPolysToLinesOff()
        clean.ConvertStripsToPolysOff()
        clean.Update()
        print(f"SurfaceNets3D mesh generated for {name}")
        poly = PolyData(clean.GetOutput())

        cluster = pyacvd.Clustering(poly)
        num_clusters = int(poly.GetNumberOfCells() * (config.voxel_resample_length / config.remesh_edge_length) ** 2 / 2)
        cluster.cluster(num_clusters)
        mesh = cluster.create_mesh()
        print(f"Uniform remeshing to edge length {config.remesh_edge_length} completed for {name}")
        if config.output_format == "stl":
            writer = vtk.vtkSTLWriter()
        else:
            writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_path.joinpath(f"{name}.{config.output_format}").as_posix())
        writer.SetInputData(mesh)
        writer.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NIFTI files")
    parser.add_argument("--input_dir", type=str, help="Root directory containing NIFTI files")
    parser.add_argument("--output_dir", type=str, help="Output directory for processed files")
    parser.add_argument("--voxel_resample_length", type=float, help="Voxel edge length after resampling")
    parser.add_argument("--closing_radius", type=int, help="Voxel radius of ball kernel used to morphological closing of labels.")
    parser.add_argument("--smoothing_distance", type=float, help="Radial distance a node can move during smoothing")
    parser.add_argument(
        "--smoothing_relaxation_factor", help="Smoothing relaxation factor. Lower is more stable but requires more iterations."
    )
    parser.add_argument("--smoothing_iterations", type=int, help="Smoothing iterations")
    parser.add_argument("--remesh_edge_length", type=float, help="Target edge length after uniform remeshing")
    parser.add_argument("--output_format", choices=["vtp", "stl"], help="Output file format")
    parser.add_argument(
        "--config_file", type=str, help="Path to configuration file (additional CLI arguments will override setting in here.)"
    )

    args = parser.parse_args()

    if args.config_file is not None:
        with open(args.config_file, "r") as f:
            run_config = Config(**json.load(f))
    else:
        run_config = Config()

    d = vars(args)
    for key, value in d.items():
        if value is not None:
            setattr(run_config, key, value)

    main(run_config)
