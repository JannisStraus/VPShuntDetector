# VPShuntDetector

VPShuntDetector is a Python project designed to detect various VPShunt valves
from X-ray images. The tool supports several valve types including:

- Codman Certas
- Codman Hakim
- Codman Uni-Shunt
- paediGAV
- proGAV Gravitationseinheit
- proGAV
- proSA

## Features

- **Valve Detection:** Identify multiple VPShunt valve types from X-ray images.
- **[Detailed Results](#output-details):** For each input image, the tool
outputs:
  - A `results.csv` file containing the confidence and predicted label for each
  of the 5 folds, along with the aggregated final confidence and prediction.
  - An output image with a bounding box overlay showing the predicted valve model.
- **[Instruction Display](#instruction-images):** Optionally, if an instruction
directory is provided, corresponding instruction images for each valve model
will be displayed alongside the output image.

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone git@github.com:JannisStraus/VPShuntDetector.git
cd VPShuntDetector
pip install -e .
```

## Usage

After installation, you can run the tool directly from the command line:

```bash
vpshuntdetector -i <input> -o <output>
```

If you have a directory with instruction images, include it using the
--instructions flag (see [Instruction Images](#instruction-images) for more
details):

```bash
vpshuntdetector -i <input> -o <output> --instructions <instructions_directory>
```

## Command Line Arguments

- `-i`, `--input`: **(Required)** Path to the input image or directory.
- `-o`, `--output`: **(Required)** Path where the output results will be saved.
- `--instructions`: _(Optional)_ Directory with instruction images for valve
models.
- `-d`, `--device`: _(Optional)_ Device to perform inference on (e.g. `cuda` or
`cpu`).

## Instruction Images

When using the `--instructions` option, the provided folder should contain one
PNG image for each supported valve type. The image files must be named exactly
as the valve type with a `.png` or `.jpg` extension. For example, the folder
should include the following files:

```bash
example_instructions/
├── Codman Certas.png
├── Codman Hakim.png
├── Codman Uni-Shunt.png
├── paediGAV.png
├── proGAV Gravitationseinheit.png
├── proGAV.png
└── proSA.png
```

## Output Details

In the output directory specified by the -o option, VPShuntDetector generates:

- **results.csv:** A CSV file containing, for every input image, the predicted
label and its confidence score for each of the 5 folds. This file also includes
the final aggregated confidence and prediction.
- **Output Images:** For every input image, an output image is saved with a
bounding box overlay that highlights the predicted valve model.
- **Instruction Display:** If an instruction directory is provided via
--instructions, the corresponding instruction image for the predicted valve
model will be displayed alongside the output image.
