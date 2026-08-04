"""
This module contains configuration classes for constructing pipelines to process segmentations and generate surface meshes.
"""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


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

    source_files: list[str | Path] = Field(default_factory=list)
    """
    List of label images.
    """
    lut: dict[str, int] | None = None
    """
    Dictionary mapping a label name to its integer value. If only 1 file is indicated for
    source_files, it is recommended to define this lut. Otherwise, label names will just
    be strings of their respective integer values.
    """
    output_path: str | Path
    """
    Path to write output files to.
    """
    processing_options: SegmentationProcessing
    """
    BaseModel defining segmentation processing options.
    """


class MmgOptions(BaseModel, frozen=True):
    hmin: float = Field(default=0.2, gt=0.0)
    """
    Minimum target element edge length
    """
    hmax: float = Field(default=1.0, gt=0.0)
    """
    Maximum target element edge length
    """
    hgrad: float = Field(default=1.5, ge=1.0)
    """
    Ratio defining how quickly edge length can change
    """
    divisions_per_circle: float = Field(default=8.0, gt=1.0)
    """
    Assuming a circle with the local radius of curvature, edge length
    is defined as arc length of circle with this many divisions
    """

    @model_validator(mode="after")
    def _check_hmax_hmin(self):
        if self.hmax < self.hmin:
            raise ValueError("hmax must be greater than or equal to hmin")
        return self


class AcvdOptions(BaseModel, frozen=True):
    edge_length: float = Field(default=1.0, gt=0.0)
    """
    Target edge length for ACVD remeshing
    """


class SurfaceMeshPipeline(BaseModel, frozen=True):
    label_file: str | Path
    """
    Path to image file containing all anatomical labels
    """
    lut_file: str | Path
    """
    Path to JSON file containing lookup table for anatomical labels
    """
    output_directory: str | Path
    """
    Path for output files
    """
    taubin_smoothing_factor1: float = 0.8
    """
    Passband for the first Taubin smoothing operation
    """
    taubin_iterations1: int = 60
    """
    Number of iterations for the first Taubin smoothing operation.
    NOTE: If <= 0, no smoothing is performed.
    """
    remesh_options: MmgOptions | AcvdOptions | None = None
    """
    BaseModel for remesh options. If None, no remesh is performed.
    """
    taubin_smoothing_factor2: float = 0.2
    """
    Passband for the Taubin smoothing operation after remesh
    """
    taubin_iterations2: int = 40
    """
    Number of iterations for the Taubin smoothing operation after remesh
    NOTE: If <= 0 or if `remesh_options` is None, no smoothing is performed.
    """
