# Summary

This is a script that converts NIfTI files to triangular meshes. It applies the following steps:

- Detects all NIfTI files in provided input_dir
- Looping over each file:
  - Reads all NIfTI files in provided input_dir as a SimpleITK image
  - Upsamples the image to an isotropic voxel size with the specified edge length: resample_voxel_length (default=0.3)
  - Performs a Morphological Closing operation with indicated kernel radius: closing_radius (default=3) to fill small holes and lessen sharp features in the upsampled label volume
- Combines all processed label volumes into a single volume
- Uses vtkSurfaceNets3D to convert the label map into triangular meshes
  - Applies constrained Laplacian smoothing based on provided parameters:
    - smoothing_iterations (default=1000)
    - smoothing_relaxation_factor (default=0.01)
    - smoothing_distance (default=0.3)
- Performs uniform remeshing of each triangular mesh targetting a specified edge length: remesh_edge_length (default=1.0)
- Saves each mesh to disk in specified format at {output_dir}/{NIfTI_file_stem}.{output_format} where the defaults are:
  - output_dir (default="output")
  - output_format (default="vtp", choices=["vtp", "stl"])
- {output_dir}/config.json is also written with all configuration parameters used during execution.



# Installation

With uv (recommended):

Install uv on Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install uv on Windows in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

From the seg2mesh directory, run:

```bash
uv sync
```

# Usage

Without activating the virtual environment, you can run the script with:

```bash
uv run main.py --help
```

to see the available options.

You'll typically need to specify at least the input_dir.

```bash
uv run main.py --input_dir /path/to/input
```

This will find all files ending with *.nii* and convert them to triangular meshes saved to
your current working directory as *[original_name].vtp*. To output as STL files, use the `--output_format` option:

```bash
uv run main.py --input_dir /path/to/input --output_format stl
```

To create an output directory, use the `--output_dir` option:

```bash
uv run main.py --input_dir /path/to/input --output_dir /path/to/output
```

The configuration settings you used for the run will be saved as *config.json* in the output directory.

To run with settings indicated in a configuration JSON file, use the `--config_file` option:

```bash
uv run main.py --config_file /path/to/config.json
```

Any CLI arguments specified in addition to a configuration file will override the settings in the file.
