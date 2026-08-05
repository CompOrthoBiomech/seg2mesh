# Summary

`seg2mesh` is a toolkit for processing segmentatation label images, extracting isocontours from label images as triangular meshes, and processing these meshes with various downstream tasks.

For label image processing, operations such as:

- Resampling

  - To adjust image resolution and/or apply transforms
  - To merge labels in different coordinate systems or resolutions

- Morphological Opening and Closing
- Island Removal
- Binary Pruning (To remove spurious label regions)
- A multistep procedure to make labels contiguous

are provided.

For mesh processing, operations such as:

- Taubin Smoothing
- Uniform Remeshing with the Approximated Discreted Centroidal Voronoi Diagram algorithm
- Adaptive Remeshing with `mmg3d` using a metric automatically assigned based on local curvature

are provided.

`seg2mesh` also enables characterization of geometric changes between processed and unprocessed meshes including:

Distance metrics:

- Hausdorff distance
- Mean symmetric surface distance
- Root mean square distance

Classification metrics:

- Dice coefficient
- Intersection over union
- Accuracy

# Getting Started

Please refer to the `seg2mesh` [Documentation](https://comporthobiomech.github.io/seg2mesh/) for
detailed instructions on installation, example usage, and API documentation auto-generated via Sphinx.
