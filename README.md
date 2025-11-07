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
