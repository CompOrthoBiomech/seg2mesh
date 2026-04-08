import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import gmsh
import numpy as np
from vtk import vtkPolyData, vtkSTLWriter, vtkXMLPolyDataReader


@dataclass
class Config:
    filepaths: list[str]
    element_order: Literal[1, 2] = 1
    element_edge_length: float = 2.0
    reduced_integration: bool = False


gmsh.initialize()


def read_vtp(file: Path) -> vtkPolyData:
    reader = vtkXMLPolyDataReader()
    reader.SetFileName(file.as_posix())
    reader.Update()
    return reader.GetOutput()


def write_stl(poly_data: vtkPolyData, file: str):
    writer = vtkSTLWriter()
    writer.SetFileName(file)
    writer.SetInputData(poly_data)
    writer.Write()


def create_geometry_and_mesh(config: Config):
    gmsh.option.set_number("Mesh.ElementOrder", config.element_order)
    gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.set_number("Mesh.Algorithm", 6)
    gmsh.option.set_number("Mesh.Algorithm3D", 10)
    gmsh.option.set_number("Mesh.MeshSizeMin", config.element_edge_length)
    gmsh.option.set_number("Mesh.MeshSizeMax", config.element_edge_length)
    gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 20)
    gmsh.option.set_number("Mesh.CompoundClassify", 0)
    if config.element_order == 2:
        gmsh.option.set_number("Mesh.HighOrderOptimize", 1)
        gmsh.option.set_number("Mesh.SecondOrderIncomplete", int(config.element_order))
    else:
        gmsh.option.set_number("Mesh.HighOrderOptimize", 0)
    for file in config.filepaths:
        gmsh.clear()
        file = Path(file)
        if file.name.endswith(".vtp"):
            poly = read_vtp(file)
            with NamedTemporaryFile(suffix=".stl") as tmp:
                write_stl(poly, tmp.name)
                gmsh.merge(tmp.name)
        else:
            gmsh.merge(file.as_posix())

        gmsh.model.mesh.classify_surfaces(40.0 * np.deg2rad(180.0), boundary=False, forReparametrization=True)
        gmsh.model.mesh.create_geometry()
        surfaces = gmsh.model.get_entities(2)
        surface_loop = gmsh.model.geo.add_surface_loop([s[1] for s in surfaces])
        gmsh.model.geo.add_volume([surface_loop])
        gmsh.model.geo.add_physical_group(3, [v[1] for v in gmsh.model.get_entities(3)])
        gmsh.model.geo.add_physical_group(2, [s[1] for s in gmsh.model.get_entities(2)], name="surface")
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(2)
        gmsh.write(f"{file.stem}.msh")
        # gmsh.model.mesh.generate(3)
        # gmsh.write(f"{file.stem}.msh")
        # gmsh.write(f"{file.stem}.vtk")
    gmsh.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = Config(**json.load(f))
    create_geometry_and_mesh(config)
