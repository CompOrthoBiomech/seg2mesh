from pathlib import Path
from sys import stderr

from loguru import logger
from polars import DataFrame
from vtkmodules.all import vtkPolyData

from seg2mesh import disk, smesh
from seg2mesh.config import AcvdOptions, MmgOptions, SurfaceMeshPipeline, parse_model_from_json, save_model_to_json
from seg2mesh.vol import NamedLabelImage, convert_segmentation_to_vtk


def main(config: SurfaceMeshPipeline):
    output_path = Path(config.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    image = disk.read_image(config.label_file)
    lut = disk.read_lut(config.lut_file)
    vtk_image = convert_segmentation_to_vtk(NamedLabelImage(image=image, lut=lut))
    isocontours = smesh.extract_isocontours(vtk_image)
    metrics = {}
    if config.calculate_classification_metrics:
        for metric in ("Dice Coefficient", "Intersection over Union", "Accuracy"):
            metrics[metric] = []
    if config.calculate_classification_metrics:
        for metric in ("Hausdorff Distance", "Mean Symmetric Surface Distance", "Root Mean Square Distance"):
            metrics[metric] = []
    for name, contour in isocontours.items():
        logger.info(f"Processing contour: {name}")
        new_contour = vtkPolyData()
        new_contour.DeepCopy(contour)
        if config.taubin_iterations1 > 0:
            new_contour = smesh.taubin_smooth(contour, config.taubin_iterations1, config.taubin_smoothing_factor1)
        if config.remesh_options is not None:
            if isinstance(config.remesh_options, MmgOptions):
                new_contour = smesh.mmg_remesh(
                    new_contour,
                    hmin=config.remesh_options.hmin,
                    hmax=config.remesh_options.hmax,
                    divisions_per_circle=config.remesh_options.divisions_per_circle,
                    hgrad=config.remesh_options.hgrad,
                )
            elif isinstance(config.remesh_options, AcvdOptions):
                new_contour = smesh.remesh(
                    new_contour,
                    edge_length=config.remesh_options.edge_length,
                )
        if config.taubin_iterations2 > 0:
            new_contour = smesh.taubin_smooth(new_contour, config.taubin_iterations2, config.taubin_smoothing_factor2)

        new_contour = smesh.compute_normals(new_contour)

        if config.calculate_distance_metrics:
            new_contour, distance_metrics = smesh.evaluate_distance_metrics(new_contour, contour)
            for metric, value in distance_metrics.items():
                metrics[metric].append(value)
        if config.calculate_classification_metrics:
            classification_metrics = smesh.evaluate_volume_metrics(new_contour, contour, voxel_edge=config.voxel_edge)
            new_contour = smesh.add_scalardict_to_field_data(classification_metrics, new_contour)
            for metric, value in classification_metrics.items():
                metrics[metric].append(value)
        for ftype in config.output_formats:
            match ftype:
                case "vtp":
                    vtp_path = output_path.joinpath("vtp")
                    if not vtp_path.exists():
                        vtp_path.mkdir(parents=True, exist_ok=True)
                    disk.write_vtp(new_contour, vtp_path.joinpath(f"{name}.vtp"))
                case "stl":
                    stl_path = output_path.joinpath("stl")
                    if not stl_path.exists():
                        stl_path.mkdir(parents=True, exist_ok=True)
                    disk.write_vtp(new_contour, stl_path.joinpath(f"{name}.stl"))
                case "ply":
                    ply_path = output_path.joinpath("ply")
                    if not ply_path.exists():
                        ply_path.mkdir(parents=True, exist_ok=True)
                    disk.write_vtp(new_contour, ply_path.joinpath(f"{name}.ply"))
                case "obj":
                    obj_path = output_path.joinpath("obj")
                    if not obj_path.exists():
                        obj_path.mkdir(parents=True, exist_ok=True)
                    disk.write_vtp(new_contour, obj_path.joinpath(f"{name}.obj"))
    save_model_to_json(config, output_path.joinpath("smesh_pipeline_config.json"))
    if metrics:
        df = DataFrame(metrics)
        df.write_csv(output_path.joinpath("error_metrics.csv"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline to generate and process surface mesh isocountours from label image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level")
    parser.add_argument("--log_file", type=str, default="smesh_pipeline.log", help="Path to the log file")
    args = parser.parse_args()
    logger.remove()
    logger.enable("seg2mesh")
    logger.add(args.log_file, level=args.log_level, mode="w")
    logger.add(stderr, level=args.log_level)

    config = parse_model_from_json(SurfaceMeshPipeline, args.config)
    main(config)
