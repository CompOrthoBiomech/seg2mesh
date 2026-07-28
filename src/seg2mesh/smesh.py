import numpy as np
from pyacvd import Clustering
from pyvista import PolyData
from vtkmodules.all import (
    vtkCleanPolyData,
    vtkDiscreteFlyingEdges3D,
    vtkDistancePolyDataFilter,
    vtkFillHolesFilter,
    vtkGeometryFilter,
    vtkPolyData,
    vtkPolyDataNormals,
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
    # Taubin smoothing
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


def _clean_poly(poly: vtkPolyData) -> vtkPolyData:
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
    pv_poly = PolyData(poly)

    # get average edge length for first poly cell
    cell = pv_poly.get_cell(0)
    avg_edge_length = np.mean([np.linalg.norm(edge.points[1] - edge.points[0]) for edge in cell.edges])
    # determine number of clusters based on initial and target edge length
    num_clusters = int(pv_poly.GetNumberOfCells() * (avg_edge_length / edge_length) ** 2 / 2)
    # Perform ACVD remesh and clean the output
    cluster = Clustering(pv_poly)
    cluster.cluster(num_clusters)
    mesh = _clean_poly(cluster.create_mesh())
    return mesh


def compute_normals(poly: vtkPolyData) -> vtkPolyData:
    normals = vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.ConsistencyOn()
    normals.SplittingOn()
    normals.Update()
    return normals.GetOutput()


def extract_isocontours(volume: NamedVTKImage):
    iso = vtkDiscreteFlyingEdges3D()
    iso.SetInputData(volume.image)
    iso.GenerateValues(len(volume.lut), 1, len(volume.lut))
    iso.Update()
    return iso.GetOutput()
