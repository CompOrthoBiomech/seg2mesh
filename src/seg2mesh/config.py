"""
This module contains configuration classes for constructing pipelines to process segmentations and generate surface meshes.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SegmentationProcessing(BaseModel, frozen=True):
    """
    Configuration class for segmentation processing.
    """

    close: bool = True
    """Perform morphological closing on segmented labels"""

    closing_radius: tuple[int, int, int] = (3, 3, 3)
    """The radius of the closing operation"""

    median_filter: bool = True
    """Median filter the segmented labels"""

    median_filter_radius: tuple[int, int, int] = (1, 1, 1)
    """The radius of the median filter"""

    open: bool = False
    """Perform morphological opening on segmented labels"""

    opening_radius: tuple[int, int, int] = (1, 1, 1)
    """The radius of the opening operation"""

    make_contiguous: list[tuple[str, str]] = Field(default_factory=list)
    """A list of tuples specifying the labels to make contiguous. First element in tuple will overwrite the second element."""

    contiguous_closing_radius: tuple[int, int, int] = (3, 3, 3)
    """The radius of the contiguous closing operation"""


class SegmentationPipeline(BaseModel, frozen=True):
    """
    Configuration class for a segmentation pipeline.
    """

    source_path: str | Path
    output_path: str | Path
    file_extension: Literal[".nii", ".nii.gz", ".nrrd", ".mha"] = ".nii"
    processing_options: SegmentationProcessing


class SurfaceMeshPipeline(BaseModel, frozen=True):
    label_file: str | Path
    output_directory: str | Path
    taubin_passband1: float = 0.001
    taubin_iterations1: int = 40
    taubin_passband2: float = 0.01
    taubin_iterations2: int = 20
