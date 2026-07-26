from pathlib import Path

from pydantic import BaseModel, Field


class SegmentationProcessing(BaseModel):
    close: bool = True
    closing_radius: tuple[int, int, int] = (3, 3, 3)
    median_filter: bool = True
    median_filter_radius: tuple[int, int, int] = (3, 3, 3)
    open: bool = True
    opening_radius: tuple[int, int, int] = (1, 1, 1)
    make_contiguous: list[tuple[str, str]] = Field(default_factory=list)


class SegmentationPipeline(BaseModel):
    source_directory: str | Path
    output_directory: str | Path
    processing_options: SegmentationProcessing


class SurfaceMeshPipeline(BaseModel):
    label_file: str | Path
    output_directory: str | Path
    taubin_passband1: float = 0.001
    taubin_iterations1: int = 40
    taubin_passband2: float = 0.01
    taubin_iterations2: int = 20
