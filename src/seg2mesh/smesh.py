from typing import Literal

import numpy as np
import numpy.typing as npt
from loguru import logger
from pyacvd import Clustering
from pyvista import PolyData
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
from vtkmodules.all import (
    VTK_CHAR,
    vtkCleanPolyData,
    vtkDataObject,
    vtkDataSetAttributes,
    vtkDiscreteFlyingEdges3D,
    vtkFillHolesFilter,
    vtkFloatArray,
    vtkGeometryFilter,
    vtkHausdorffDistancePointSetFilter,
    vtkImageData,
    vtkImageMathematics,
    vtkImageStencil,
    vtkImplicitPolyDataDistance,
    vtkPolyData,
    vtkPolyDataNormals,
    vtkPolyDataToImageStencil,
    vtkThreshold,
    vtkUnstructuredGrid,
    vtkWindowedSincPolyDataFilter,
)
from vtkmodules.util.numpy_support import vtk_to_numpy

from .vol import NamedVTKImage


def evaluate_polydata_distance(poly1: vtkPolyData, poly2: vtkPolyData):
    distance_filter = vtkImplicitPolyDataDistance()
    distances = vtkFloatArray()
    distances.SetName("Signed Distances")
    distances.SetNumberOfValues(poly2.GetNumberOfPoints())
    distances.SetNumberOfComponents(1)
    distance_filter.SetInput(poly1)
    for i in range(poly2.GetNumberOfPoints()):
        point = poly2.GetPoint(i)
        distances.InsertValue(i, distance_filter.EvaluateFunction(*point))
    poly2.GetPointData().AddArray(distances)

    return poly2


def get_compatible_origin_and_size(poly1: vtkPolyData, poly2: vtkPolyData, voxel_edge: float) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Given two vtkPolyData objects, returns the compatible origin and size to create corresponding voxelizations.

    :param poly1: The first vtkPolyData object.
    :param poly2: The second vtkPolyData object.
    :param voxel_edge: The edge length of the voxels.

    :returns: The compatible origin and size as a tuple of numpy arrays.
    """
    bbox1 = np.array(poly1.GetBounds())
    bbox2 = np.array(poly2.GetBounds())
    origin = np.min([bbox1[0::2], bbox2[0::2]], axis=0)
    corner = np.max([bbox1[1::2], bbox2[1::2]], axis=0)
    size = np.ceil((corner - origin) / voxel_edge).astype(int)
    return origin, size


def voxelize_mesh(
    mesh: vtkPolyData, voxel_edge: float, origin: npt.NDArray, size: npt.NDArray, max_voxels: int = 1_000_000_000
) -> vtkImageData:
    """
    Voxelizes a mesh into a vtkImageData volume using the utilizes a vtkPolyDataToImageStencil filter.

    :param voxel_edge: The edge length of the voxels.
    :param origin: The origin of the vtkImageData volume.
    :param size: The size ([x, y, z] in voxels) of the vtkImageData volume.
    :param max_voxels: The maximum number of voxels allowed, to avoid excessive memory usage.

    :return: The vtkImageData volume.
    """
    total_voxels = int(np.prod(size))
    logger.info(f"...Voxelizing PolyData with {voxel_edge=} for {total_voxels=}")
    if total_voxels > max_voxels:
        message = f"......To achieve {voxel_edge=} requires {total_voxels} voxels, which is greater than {max_voxels=}"
        logger.error(message)
        raise RuntimeError(message)

    image = vtkImageData()
    image.SetSpacing([voxel_edge] * 3)
    image.SetDimensions(*size)
    image.SetExtent([0, size[0] - 1, 0, size[1] - 1, 0, size[2] - 1])
    image.SetOrigin(*origin)
    image.AllocateScalars(VTK_CHAR, 1)
    image.GetPointData().GetScalars().Fill(1)
    image.Modified()

    poly2stencil = vtkPolyDataToImageStencil()
    poly2stencil.SetInputData(mesh)
    poly2stencil.SetOutputOrigin(image.GetOrigin())
    poly2stencil.SetOutputSpacing(image.GetSpacing())
    poly2stencil.SetOutputWholeExtent(image.GetExtent())
    poly2stencil.Update()

    stencil = vtkImageStencil()
    stencil.SetInputData(image)
    stencil.SetStencilConnection(poly2stencil.GetOutputPort())
    stencil.ReverseStencilOff()
    stencil.SetBackgroundValue(0)
    stencil.Update()

    return stencil.GetOutput()


def image_boolean(image1: vtkImageData, image2: vtkImageData, operation: Literal["Union", "Intersection", "Difference"]) -> vtkImageData:
    boolean = vtkImageMathematics()
    boolean.SetInput1Data(image1)
    boolean.SetInput2Data(image2)
    if operation == "Union":
        boolean.SetOperationToMax()
    elif operation == "Intersection":
        boolean.SetOperationToMultiply()
    elif operation == "Difference":
        boolean.SetOperationToMin()
    boolean.Update()
    return boolean.GetOutput()


def evaluate_volume_metrics(poly1: vtkPolyData, poly2: vtkPolyData, voxel_edge: float) -> dict[str, float]:
    """
    Calculate classification scores based on voxelized volumes.

    :param poly1: The current mesh to compare.
    :param poly2: The reference mesh to compare against.
    :param voxel_edge: The edge length of the voxels used for voxelization.

    :returns: A dictionary of volume metrics including Dice Coefficient, Intersection over Union, and Accuracy.
    """
    logger.info("Evaluating classification scores")
    origin, size = get_compatible_origin_and_size(poly1, poly2, voxel_edge)
    vol1 = vtk_to_numpy(voxelize_mesh(poly1, voxel_edge, origin, size).GetPointData().GetScalars()).ravel()
    vol2 = vtk_to_numpy(voxelize_mesh(poly2, voxel_edge, origin, size).GetPointData().GetScalars()).ravel()

    metrics = {}
    metrics["Dice Coefficient"] = f1_score(vol2, vol1)
    metrics["Intersection over Union"] = jaccard_score(vol2, vol1)
    metrics["Accuracy"] = accuracy_score(vol2, vol1)
    for k, v in metrics.items():
        logger.info(f"...{k}: {v}")

    return metrics


def evaluate_distance_metrics(poly1: vtkPolyData, poly2: vtkPolyData) -> tuple[vtkPolyData, dict[str, float]]:
    """
    Calculates distance error metrics including Hausdorff distance, Mean Symmetric Surface Distance, and Root Mean Square Distance.

    :param poly1: The current mesh to compare.
    :param poly2: The reference mesh to compare against.

    :returns: A tuple containing PolyData with poly1 topology with shortest (point-cell calculated) distances from poly1 to poly2 stored on
    vertices and metrics stored as FieldData and a dictionary of metrics. NOTE: The vertex distances are only poly1 to poly2 distances, but metrics
    are calculated using both poly1 to poly2 and poly2 to poly1 distances.

    """
    logger.info("Evaluating Distance Metrics")
    distance = vtkHausdorffDistancePointSetFilter()
    distance.SetInputData(0, poly1)
    distance.SetInputData(1, poly2)
    distance.SetTargetDistanceMethodToPointToCell()
    distance.Update()

    distance1 = distance.GetOutput(0)
    distance2 = distance.GetOutput(1)

    total_points = distance2.GetNumberOfPoints() + distance2.GetNumberOfPoints()
    distance1_array = vtk_to_numpy(distance1.GetPointData().GetArray("Distance"))
    distance2_array = vtk_to_numpy(distance2.GetPointData().GetArray("Distance"))
    mssd = (np.sum(distance1_array) + np.sum(distance2_array)) / total_points
    mssd_array = vtkFloatArray()
    mssd_array.InsertNextValue(mssd)
    mssd_array.SetName("Mean Symmetric Surface Distance")
    distance2.GetFieldData().AddArray(mssd_array)
    rmse = np.sqrt((np.sum(distance1_array**2) + np.sum(distance2_array**2)) / total_points)
    rmse_array = vtkFloatArray()
    rmse_array.InsertNextValue(rmse)
    rmse_array.SetName("Root Mean Square Distance")
    distance2.GetFieldData().AddArray(rmse_array)

    metrics = {
        "Hausdorff Distance": distance2.GetFieldData().GetArray("HausdorffDistance").GetValue(0),
        "Mean Symmetric Surface Distance": mssd,
        "Root Mean Square Distance": rmse,
    }
    for k, v in metrics.items():
        logger.info(f"...{k}: {v}")

    return distance2, metrics


def taubin_smooth(poly: vtkPolyData, iterations: int = 40, passband: float = 0.01) -> vtkPolyData:
    """
    Apply Taubin smoothing to the input vtkPolyData.

    :param poly: The input vtkPolyData to be smoothed.
    :param iterations: The number of smoothing iterations. More iterations result in smoother output.
    :param passband: The passband parameter for the smoothing filter. Lower values result in smoother output.
    :return: The smoothed vtkPolyData.
    """
    smooth = vtkWindowedSincPolyDataFilter()
    smooth.SetInputData(poly)
    smooth.SetNumberOfIterations(iterations)
    smooth.SetPassBand(passband)
    smooth.BoundarySmoothingOff()
    smooth.FeatureEdgeSmoothingOff()
    smooth.NonManifoldSmoothingOn()
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    return smooth.GetOutput()


def _grid_to_poly(grid: vtkUnstructuredGrid) -> vtkPolyData:
    geo = vtkGeometryFilter()
    geo.SetInputData(grid)
    geo.Update()
    return geo.GetOutput()


def clean_poly(poly: vtkPolyData) -> vtkPolyData:
    """
    Clean the input vtkPolyData using vtkCleanPolyData and fill any holes using vtkFillHolesFilter.

    :param poly: The input vtkPolyData to be cleaned.
    :return: The cleaned vtkPolyData with holes filled.
    """
    clean = vtkCleanPolyData()
    clean.SetInputData(poly)
    clean.Update()

    tri = vtkPolyData()
    tri.SetPoints(clean.GetOutput().GetPoints())
    tri.SetPolys(clean.GetOutput().GetPolys())

    fillholes = vtkFillHolesFilter()
    fillholes.SetInputData(tri)
    fillholes.SetHoleSize(1e9)
    fillholes.Update()
    return fillholes.GetOutput()


def remesh(poly: vtkPolyData, edge_length: float = 1.0) -> vtkPolyData:
    """
    Remesh the input vtkPolyData using ACVD (Approximated Centroidal Voronoi Diagrams).

    :param poly: The input vtkPolyData to be remeshed.
    :param edge_length: The target edge length for the remeshed mesh.
    :return: The remeshed vtkPolyData.
    """
    logger.info("...Remeshing polydata using ACVD")
    pv_poly = PolyData(poly)
    # get average cell edge length
    edges = pv_poly.extract_feature_edges(boundary_edges=False, feature_edges=False, manifold_edges=True, non_manifold_edges=False)
    edges = edges.compute_cell_sizes(length=True, area=False, volume=False)
    avg_edge_length = edges.cell_data["Length"].mean()
    # determine number of clusters based on initial and target edge length
    num_clusters = int(pv_poly.GetNumberOfCells() * (avg_edge_length / edge_length) ** 2 / 2)
    logger.debug(f"......Average edge length: {avg_edge_length}")
    logger.debug(f"......{num_clusters} clusters estimated to achieve target edge length: {edge_length}")
    # Perform ACVD remesh and clean the output
    cluster = Clustering(pv_poly)
    cluster.cluster(num_clusters)
    mesh = clean_poly(cluster.create_mesh())
    return mesh


def compute_normals(poly: vtkPolyData) -> vtkPolyData:
    """
    Compute vertex normals for the input vtkPolyData using vtkPolyDataNormals.
    Normal consistency and splitting at feature edges are enabled.

    :param poly: The input vtkPolyData to compute normals for.
    :return: The vtkPolyData with computed normals.
    """
    normals = vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOn()
    normals.Update()
    return normals.GetOutput()


def extract_isocontours(volume: NamedVTKImage) -> dict[str, vtkPolyData]:
    """
    Triangulate isocontours from vtkImageData using vtkDiscreteFlyingEdges3D,
    extract using vtkThreshold, clean, and add to dictionary by name from the LUT.

    :param volume: Contains vtkImage Data, `image`, and lookup table, `lut`, mapping image intensity values to contour names.
    :return: A dictionary of isocontours, where the keys are contour names and the values are vtkPolyData.
    """
    iso = vtkDiscreteFlyingEdges3D()
    iso.SetInputData(volume.image)
    iso.GenerateValues(len(volume.lut), 1, len(volume.lut))
    iso.Update()
    all_contours = iso.GetOutput()

    contours = {}
    for name, label_id in volume.lut.items():
        logger.info(f"Extracting isocountour for {name}")
        threshold = vtkThreshold()
        threshold.SetInputData(all_contours)
        threshold.SetInputArrayToProcess(0, 0, 0, vtkDataObject.FIELD_ASSOCIATION_POINTS, vtkDataSetAttributes.SCALARS)
        threshold.SetThresholdFunction(vtkThreshold.THRESHOLD_BETWEEN)
        threshold.SetLowerThreshold(label_id)
        threshold.SetUpperThreshold(label_id)
        threshold.Update()

        contours[name] = clean_poly(_grid_to_poly(threshold.GetOutput()))
    return contours
