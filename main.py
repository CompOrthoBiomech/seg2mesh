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
    make_congruent: list[tuple[str, str]] | None = None
    smoothing_iterations: int = 40
    smoothing_passband: float = 0.01
    remesh_edge_length: float = 0.5
    smoothing2_iterations: int = 0
    smoothing2_passband: float = 0.01
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


def taubin_smooth(poly: vtk.vtkPolyData, iterations: int = 40, passband: float = 0.01) -> vtk.vtkPolyData:
    # Taubin smoothing
    smooth = vtk.vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(poly)
    smooth.SetNumberOfIterations(iterations)
    smooth.SetPassBand(passband)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.SetGenerateErrorScalars(1)
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    return smooth.GetOutput()


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


def make_congruent(label1: sitk.Image, label2: sitk.Image) -> sitk.Image:
    union = sitk.Or(label1, label2)
    union = sitk.GrayscaleMorphologicalClosing(union, [3, 3, 3])
    dilate_label1 = sitk.BinaryDilate(label1, [3, 3, 3])
    dilate_label2 = sitk.BinaryDilate(label2, [3, 3, 3])
    filled_label1 = sitk.Or(label1, sitk.And(dilate_label1, dilate_label2))
    filled_label1 *= union
    filled_label1 = sitk.BinaryMedian(filled_label1, [1, 1, 1]) * sitk.Not(label2)
    return filled_label1


def sitk_to_vtk_image(image: sitk.Image) -> vtk.vtkImageData:
    nparray = sitk.GetArrayFromImage(image)
    vtk_data = numpy_support.numpy_to_vtk(nparray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtkimage = vtk.vtkImageData()
    vtkimage.SetDimensions(nparray.shape[2], nparray.shape[1], nparray.shape[0])
    vtkimage.SetSpacing(image.GetSpacing())
    vtkimage.SetOrigin(image.GetOrigin())
    vtkimage.GetPointData().SetScalars(vtk_data)
    return vtkimage


def merge_labels(label1: sitk.Image, label2: sitk.Image) -> sitk.Image:
    union = sitk.Or(label1, label2)
    union = sitk.GrayscaleMorphologicalClosing(union, [3, 3, 3])
    union = sitk.BinaryMedian(sitk.BinaryFillhole(union))
    label2_vtk = sitk_to_vtk_image(label2)
    union = sitk_to_vtk_image(union)
    meshes = []
    for vol in (label2_vtk, union):
        snets = vtk.vtkDiscreteFlyingEdges3D()
        snets.SetInputData(vol)
        snets.GenerateValues(1, 1, 1)
        snets.Update()
        log.info("Peforming Taubin Smoothing")
        mesh = taubin_smooth(snets.GetOutput(), 80, 0.001)
        meshes.append(mesh)
    clip_function = vtk.vtkImplicitPolyDataDistance()
    clip_function.SetInput(meshes[0])

    clip = vtk.vtkClipPolyData()
    clip.SetClipFunction(clip_function)
    clip.SetValue(0.05)
    clip.SetInputData(meshes[1])

    connected = vtk.vtkPolyDataConnectivityFilter()
    connected.SetInputConnection(clip.GetOutputPort())
    connected.SetExtractionModeToLargestRegion()

    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(connected.GetOutputPort())
    smooth.SetSourceData(meshes[0])
    smooth.BoundarySmoothingOn()
    smooth.SetNumberOfIterations(1000)
    smooth.Update()
    return smooth.GetOutput()


def main(config: Config):
    output_path = Path(config.output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    original_volumes = []
    volume_names = []
    largest_image = None
    for i, file in enumerate(Path(config.input_dir).glob("*.nii")):
        img = sitk.ReadImage(file.as_posix(), outputPixelType=sitk.sitkUInt8)
        if largest_image is None or np.prod(largest_image.GetSize()) < np.prod(img.GetSize()):
            largest_image = img
        original_volumes.append(img)
        volume_names.append(file.stem)
    global_image_origin = largest_image.GetOrigin()
    global_image_direction = largest_image.GetDirection()
    scale = [s / config.voxel_resample_length for s in largest_image.GetSpacing()]
    target_dim = [int(s * d + 0.5) for (s, d) in zip(scale, largest_image.GetSize())]
    volumes = []
    volume_lut = {}
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
        label = sitk.BinaryFillhole(label)
        label = sitk.BinaryMedian(label, [2, 2, 2])
        label = sitk.GrayscaleMorphologicalOpening(label, [1, 1, 1])
        label = sitk.Resample(label, interpolator=sitk.sitkNearestNeighbor, referenceImage=upsampled)
        volumes.append(label)
        log.info(f"Added resampled {volume_name} to volumes")
        volume_lut[volume_name] = i
    if config.make_congruent is not None:
        for (
            volume1,
            volume2,
        ) in config.make_congruent:
            log.info(f"Making {volume1} and {volume2} congruent. {volume1} will overwrite")
            volumes[volume_lut[volume1]] = make_congruent(volumes[volume_lut[volume1]], volumes[volume_lut[volume2]])
            mesh = merge_labels(volumes[volume_lut[volume1]], volumes[volume_lut[volume2]])
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(f"{volume1}_clipped.vtp")
            writer.SetInputData(mesh)
            writer.Write()

    composite = volumes[0]
    for i, volume in enumerate(volumes[1:]):
        composite += volume * (i + 2)
        composite[composite > (i + 2)] = i + 2
    composite = sitk.ConstantPad(composite, padLowerBound=(1, 1, 1), padUpperBound=(1, 1, 1), constant=0)

    sitk.WriteImage(composite, output_path.joinpath("composite.nii"))
    nparray = sitk.GetArrayFromImage(composite)
    vtk_data = numpy_support.numpy_to_vtk(nparray.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtkimage = vtk.vtkImageData()
    vtkimage.SetDimensions(nparray.shape[2], nparray.shape[1], nparray.shape[0])
    vtkimage.SetSpacing(composite.GetSpacing())
    vtkimage.SetOrigin(composite.GetOrigin())
    vtkimage.GetPointData().SetScalars(vtk_data)

    with open(output_path.joinpath("config.json"), "w") as f:
        json.dump(asdict(config), fp=f, indent=4)

    log.info("Performing FlyingEdges3D isocontouring on all labels")
    snets = vtk.vtkDiscreteFlyingEdges3D()
    snets.SetInputData(vtkimage)
    snets.GenerateValues(len(volumes), 1, len(volumes))
    snets.Update()
    snets_mesh = snets.GetOutput()

    for i, name in enumerate(volume_names):
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(snets_mesh)
        threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, vtk.vtkDataSetAttributes.SCALARS)
        threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
        threshold.SetLowerThreshold(i + 0.5)
        threshold.SetUpperThreshold(i + 1.5)
        threshold.Update()

        mesh = _clean_poly(_grid_to_poly(threshold.GetOutput()))

        log.info(f"Extracted Mesh for {name}")
        if config.smoothing_iterations > 0:
            log.info(f"Peforming Taubin Smoothing on {name}")
            mesh = taubin_smooth(mesh, config.smoothing_iterations, config.smoothing_passband)

        # Remesh using Approximated-discrete Centroidal Voronoi Diagram (ACVD) algorithm
        if config.remesh_edge_length > 0.0:
            poly = PolyData(mesh)
            cluster = Clustering(poly)
            num_clusters = int(poly.GetNumberOfCells() * (config.voxel_resample_length / config.remesh_edge_length) ** 2 / 2)
            cluster.cluster(num_clusters)
            mesh = cluster.create_mesh()
            log.info(f"Uniform remeshing to edge length {config.remesh_edge_length} completed for {name}")
            mesh = _clean_poly(mesh)
            if config.smoothing2_iterations > 0:
                log.info(f"Peforming Second Pass of Taubin Smoothing on {name}")
                mesh = taubin_smooth(mesh, config.smoothing2_iterations, config.smoothing2_passband)
        if config.output_format == "stl":
            writer = vtk.vtkSTLWriter()
        else:
            writer = vtk.vtkXMLPolyDataWriter()

        fix_normals = vtk.vtkPolyDataNormals()
        fix_normals.SetInputData(mesh)
        fix_normals.ConsistencyOn()
        fix_normals.SplittingOn()
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
    parser.add_argument("--smoothing_iterations", type=int, help="Number of Taubin smoothing iterations")
    parser.add_argument("--smoothing_passband", type=int, help="Windowed sinc function passband. Lower results in more smoothing.")
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
