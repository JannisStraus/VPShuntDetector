# VPShuntDetector

VPShuntDetector is a Python project designed to detect various VPShunt valves from X-ray images. The tool supports several valve types including:

- Codman Certas
- Codman Hakim
- Codman Uni-Shunt
- paediGAV
- proGAV Gravitationseinheit
- proGAV
- proSA

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

## Command Line Arguments
- `-i`, `--input`: **(Required)** Path to the input image or directory.
- `-o`, `--output`: **(Required)** Path where the output results will be saved.
- `--instructions`: _(Optional)_ Directory with instruction images for valve models.
- `-d`, `--device`: _(Optional)_ Device to perform inference on (e.g. `cuda` or `cpu`).

## Reference Images Folder Structure

When using the `--instructions` option, the provided folder should contain one PNG image for each supported valve type. The image files must be named exactly as the valve type with a `.png` or `.jpg` extension. For example, the folder should include the following files:

```
example_instructions/
├── Codman Certas.png
├── Codman Hakim.png
├── Codman Uni-Shunt.png
├── paediGAV.png
├── proGAV Gravitationseinheit.png
├── proGAV.png
└── proSA.png
```
