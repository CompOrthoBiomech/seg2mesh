from loguru import logger
from pyacvd import Clustering
from pyvista import PolyData
from vtkmodules.all import (
    vtkCleanPolyData,
    vtkDataObject,
    vtkDataSetAttributes,
    vtkDiscreteFlyingEdges3D,
    vtkDistancePolyDataFilter,
    vtkFillHolesFilter,
    vtkGeometryFilter,
    vtkPolyData,
    vtkPolyDataNormals,
    vtkThreshold,
    vtkUnstructuredGrid,
    vtkWindowedSincPolyDataFilter,
)

from .vol import NamedVTKImage


def evaluate_polydata_distance(poly1: vtkPolyData, poly2: vtkPolyData):
    distance_filter = vtkDistancePolyDataFilter()
    distance_filter.SetInputData(0, poly1)
    distance_filter.SetInputData(1, poly2)
    distance_filter.SignedDistanceOff()
    distance_filter.Update()
    return distance_filter.GetOutput()


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
    smooth.SetGenerateErrorScalars(1)
    smooth.NormalizeCoordinatesOn()
    smooth.Update()
    mesh = smooth.GetOutput()
    mesh.GetPointData().GetScalars().SetName("Offset")
    return mesh


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
