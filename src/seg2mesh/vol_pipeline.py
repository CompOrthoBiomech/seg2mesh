from pathlib import Path
from sys import stderr

from loguru import logger

from seg2mesh.config import SegmentationPipeline, parse_model_from_json, save_model_to_json
from seg2mesh.disk import read_image, read_images, write_image, write_lut
from seg2mesh.vol import (
    NamedLabelImage,
    create_canvas_for_volumes,
    create_lut_from_label_image,
    process_segmentation,
    resample_label_image,
    resample_volumes_to_canvas,
)


def main(config: SegmentationPipeline):
    output_path = Path(config.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    if len(config.source_files) > 1:
        volumes = read_images(filepaths=config.source_files)
        canvas = create_canvas_for_volumes(volumes, spacing=config.target_voxel_size)
        composite = resample_volumes_to_canvas(volumes, canvas)
    else:
        volume = read_image(config.source_files[0])
        image = resample_label_image(volume, spacing=config.target_voxel_size)
        if config.lut is None:
            lut = create_lut_from_label_image(image)
        else:
            lut = config.lut
        composite = NamedLabelImage(image=image, lut=lut)

    write_image(composite.image, output_path.joinpath("unprocessed.seg.nrrd"))
    processed = process_segmentation(composite, config.processing_options)

    write_image(processed.image, output_path.joinpath("processed.seg.nrrd"))
    write_lut(processed.lut, output_path.joinpath("processed.seg.json"))
    save_model_to_json(config, output_path.joinpath("vol_pipeline_config.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process a label volumes using the pre-defined pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("config", type=str, help="Path to the configuration file")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level")
    parser.add_argument("--log_file", type=str, default="vol_pipeline.log", help="Path to the log file")
    args = parser.parse_args()
    logger.remove()
    logger.enable("seg2mesh")
    logger.add(args.log_file, level=args.log_level, format="{time} {level} {message}", mode="w")
    logger.add(stderr, level=args.log_level, format="{time} {level} {message}")

    config = parse_model_from_json(SegmentationPipeline, args.config)
    main(config)
