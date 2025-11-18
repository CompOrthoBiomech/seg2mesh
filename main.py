import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import SimpleITK as sitk
import vtkmodules.all as vtk
from pyacvd import Clustering
from pyvista import PolyData
from vtkmodules.util import numpy_support

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    input_dir: str = "."
    voxel_resample_length: float = 0.2
    closing_radius: int = 5
    lap_smoothing_distance: float = 0.2
    lap_smoothing_relaxation_factor: float = 0.5
    lap_smoothing_iterations: int = 200
    taubin_smoothing_iterations: int = 40
    taubin_smoothing_passband: float = 0.01
    remesh_edge_length: float = 0.5
    output_dir: str = "output"
    output_format: Literal["vtp", "stl"] = "vtp"


def evaluate_polydata_distance(poly1: vtk.vtkPolyData, poly2: vtk.vtkPolyData):
    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.SetInputData(0, poly1)
    distance_filter.SetInputData(1, poly2)
    distance_filter.SignedDistanceOff()
    distance_filter.Update()
    return distance_filter.GetOutput()


def remove_islands(img: sitk.Image) -> sitk.Image:
    components = sitk.ConnectedComponentImageFilter()
    components.SetFullyConnected(True)
    connected = components.Execute(img)
    sorted_labels = sitk.RelabelComponent(connected, sortByObjectSize=True)
    return sorted_labels == 1


def _grid_to_poly(grid: vtk.vtkUnstructuredGrid) -> vtk.vtkPolyData:
    geo = vtk.vtkGeometryFilter()
    geo.SetInputData(grid)
    geo.Update()
    return geo.GetOutput()


def _clean_poly(poly: vtk.vtkPolyData) -> vtk.vtkPolyData:
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()

    # vtkTriangleFilter was either creating new lines or not respecting PassLinesOff
    # just create a new vtkPolyData object instead
    tri = vtk.vtkPolyData()
    tri.SetPoints(clean.GetOutput().GetPoints())
    tri.SetPolys(clean.GetOutput().GetPolys())

    fillholes = vtk.vtkFillHolesFilter()
    fillholes.SetInputData(tri)
    fillholes.SetHoleSize(1e9)
    fillholes.Update()
    return fillholes.GetOutput()


def main(config: Config):
    output_path = Path(config.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    original_volumes = []
    volume_names = []
    largest_image = None
    for i, file in enumerate(Path(config.input_dir).glob("*.nii")):
        img = sitk.ReadImage(file.as_posix(), outputPixelType=sitk.sitkUInt8)
        if largest_image is None:
            largest_image = img
        elif np.prod(largest_image.GetSize()) < np.prod(img.GetSize()):
            largest_image = img
        original_volumes.append(img)
        volume_names.append(file.stem)
    global_image_origin = largest_image.GetOrigin()
    global_image_direction = largest_image.GetDirection()
    scale = [s / config.voxel_resample_length for s in largest_image.GetSpacing()]
    target_dim = [int(s * d + 0.5) for (s, d) in zip(scale, largest_image.GetSize())]
    volumes = []
    for i, (volume_name, img) in enumerate(zip(volume_names, original_volumes)):
        upsampled = sitk.Resample(
            img,
            target_dim,
            transform=sitk.Transform(),
            interpolator=sitk.sitkNearestNeighbor,
            outputOrigin=global_image_origin,
            outputSpacing=[config.voxel_resample_length] * 3,
            outputDirection=global_image_direction,
        )
        padded = sitk.ConstantPad(
            upsampled, padLowerBound=[config.closing_radius] * 3, padUpperBound=[config.closing_radius] * 3, constant=0
        )
        labelstat = sitk.LabelStatisticsImageFilter()
        labelstat.Execute(padded, padded)
        bounds = labelstat.GetBoundingBox(1)
        size = [bounds[i] - bounds[j] + (config.closing_radius * 2) for i, j in zip([1, 3, 5], [0, 2, 4])]
        index = [bounds[i] - config.closing_radius for i in [0, 2, 4]]
        roi = sitk.RegionOfInterest(padded, size=size, index=index)

        label = sitk.GrayscaleMorphologicalClosing(roi, [config.closing_radius] * 3)
        label = sitk.Resample(roi, interpolator=sitk.sitkNearestNeighbor, referenceImage=upsampled)
        label = remove_islands(label) * (i + 1)
        volumes.append(label)
        log.info(f"Added resampled {volume_name} to volumes")

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

    with open(output_path.joinpath("config.json"), "w") as f:
        json.dump(asdict(config), fp=f, indent=4)

    # SurfaceNets3D isocontouring
    log.info("Performing SurfaceNets3D isocontouring on all labels")
    snets = vtk.vtkSurfaceNets3D()
    snets.SetInputData(vtkimage)
    snets.GenerateLabels(len(volumes), 1, len(volumes))
    # Internal constrained Laplacian smoothing of SurfaceNets3D, this constrains
    # nodes on shared boundaries of labels
    if config.lap_smoothing_iterations <= 0:
        snets.SmoothingOff()
    else:
        snets.GetSmoother().SetNumberOfIterations(config.lap_smoothing_iterations)
        snets.GetSmoother().SetConstraintDistance(config.lap_smoothing_distance)
        snets.GetSmoother().SetRelaxationFactor(config.lap_smoothing_relaxation_factor)
        snets.OptimizedSmoothingStencilsOn()
    snets.Update()
    snets_mesh = snets.GetOutput()

    for i, name in enumerate(volume_names):
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(snets_mesh)
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "BoundaryLabels")
        threshold.SetSelectedComponent(0)
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
        threshold.SetLowerThreshold(i + 0.5)
        threshold.SetUpperThreshold(i + 1.5)
        threshold.Update()

        threshold2 = vtk.vtkThreshold()
        threshold2.SetInputData(snets_mesh)
        threshold2.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "BoundaryLabels")
        threshold2.SetSelectedComponent(1)
        threshold2.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
        threshold2.SetLowerThreshold(i + 0.5)
        threshold2.SetUpperThreshold(i + 1.5)
        threshold2.Update()

        log.info(f"Extracted Mesh for {name}")
        if threshold2.GetOutput().GetNumberOfCells() > 0:
            merge = vtk.vtkAppendPolyData()
            merge.AddInputData(_grid_to_poly(threshold.GetOutput()))
            merge.AddInputData(_grid_to_poly(threshold2.GetOutput()))
            merge.Update()

            mesh = _clean_poly(merge.GetOutput())
        else:
            mesh = _clean_poly(_grid_to_poly(threshold.GetOutput()))

        if config.taubin_smoothing_iterations > 0:
            log.info(f"Peforming Taubin Smoothing on {name}")
            # Taubin smoothing
            smooth = vtk.vtkWindowedSincPolyDataFilter()
            smooth.SetInputData(mesh)
            smooth.SetNumberOfIterations(config.taubin_smoothing_iterations)
            smooth.SetPassBand(config.taubin_smoothing_passband)
            smooth.BoundarySmoothingOff()
            smooth.FeatureEdgeSmoothingOff()
            smooth.NonManifoldSmoothingOn()
            smooth.SetGenerateErrorScalars(1)
            smooth.NormalizeCoordinatesOn()
            smooth.Update()
            mesh = smooth.GetOutput()

        # Remesh using Approximated-discrete Centroidal Voronoi Diagram (ACVD) algorithm
        if config.remesh_edge_length > 0.0:
            poly = PolyData(mesh)
            cluster = Clustering(poly)
            num_clusters = int(poly.GetNumberOfCells() * (config.voxel_resample_length / config.remesh_edge_length) ** 2 / 2)
            cluster.cluster(num_clusters)
            mesh = cluster.create_mesh()
            log.info(f"Uniform remeshing to edge length {config.remesh_edge_length} completed for {name}")
        if config.output_format == "stl":
            writer = vtk.vtkSTLWriter()
        else:
            writer = vtk.vtkXMLPolyDataWriter()

        fix_normals = vtk.vtkPolyDataNormals()
        fix_normals.SetInputData(mesh)
        fix_normals.ConsistencyOn()
        fix_normals.AutoOrientNormalsOn()
        fix_normals.Update()

        writer.SetFileName(output_path.joinpath(f"{name}.{config.output_format}").as_posix())
        writer.SetInputData(fix_normals.GetOutput())
        writer.Write()
        log.info(f"Saved mesh as {name}.{config.output_format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NIFTI files", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, help="Root directory containing NIFTI files")
    parser.add_argument("--output_dir", type=str, help="Output directory for processed files")
    parser.add_argument("--voxel_resample_length", type=float, help="Voxel edge length after resampling")
    parser.add_argument("--closing_radius", type=int, help="Voxel radius of ball kernel used to morphological closing of labels.")
    parser.add_argument("--lap_smoothing_distance", type=float, help="Radial distance a node can move during Laplacian smoothing")
    parser.add_argument(
        "--lap_smoothing_relaxation_factor",
        help="Constrained Laplacian smoothing relaxation factor. Lower is more stable but requires more iterations.",
    )
    parser.add_argument("--lap_smoothing_iterations", type=int, help="Number of constrained Laplacian smoothing iterations")
    parser.add_argument("--taubin_smoothing_iterations", type=int, help="Number of Taubin smoothing iterations")
    parser.add_argument("--taubin_smoothing_passband", type=int, help="Windowed sinc function passband. Lower results in more smoothing.")
    parser.add_argument(
        "--remesh_edge_length", type=float, help="Target edge length after uniform remeshing. If negative, no remeshing is performed."
    )
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
