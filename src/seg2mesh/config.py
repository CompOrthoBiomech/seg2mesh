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

    open_list: list[str] = Field(default_factory=list)
    """A list of label 'Short Name' to perform morphological opening on"""

    opening_radius: tuple[int, int, int] = (1, 1, 1)
    """The radius of the opening operation"""

    spur_removal_length: int = 10
    """The length of spurs (in voxels) to remove"""

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
    """
    Path to image file containing all anatomical labels
    """
    output_directory: str | Path
    """
    Path for output files
    """
    taubin_passband1: float = 0.001
    """
    Passband for the first Taubin smoothing operation
    """
    taubin_iterations1: int = 40
    """
    Number of iterations for the first Taubin smoothing operation.
    NOTE: If <= 0, no smoothing is performed.
    """
    remesh_edge_length: float = -1.0
    """
    Edge length for the remesh operation.
    NOTE: If < 0, no remesh is performed.
    """
    taubin_passband2: float = 0.01
    """
    Passband for the Taubin smoothing operation after remesh
    """
    taubin_iterations2: int = 20
    """
    Number of iterations for the Taubin smoothing operation after remesh
    NOTE: If <= 0 or if `remesh_edge_length` is < 0, no smoothing is performed.
    """
