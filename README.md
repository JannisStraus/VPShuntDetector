# VPShuntDetector

[![DOI](https://img.shields.io/badge/DOI-10.1038/s41598--025--29201--0-blue)](https://doi.org/10.1038/s41598-025-29201-0)
[![Python](https://img.shields.io/badge/Python-3.10–3.13-3776AB?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDEuMS8vRU4iICJodHRwOi8vd3d3LnczLm9yZy9HcmFwaGljcy9TVkcvMS4xL0RURC9zdmcxMS5kdGQiPjxzdmcgdmVyc2lvbj0iMS4xIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOmNjPSJodHRwOi8vd2ViLnJlc291cmNlLm9yZy9jYy8iIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyIgeG1sbnM6c3ZnPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeD0iMHB4IiB5PSIwcHgiIHdpZHRoPSIxMTBweCIgaGVpZ2h0PSIxMTBweCIgdmlld0JveD0iMC4yMSAtMC4wNzcgMTEwIDExMCIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwLjIxIC0wLjA3NyAxMTAgMTEwIiB4bWw6c3BhY2U9InByZXNlcnZlIj48bGluZWFyR3JhZGllbnQgaWQ9IlNWR0lEXzFfIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeDE9IjYzLjgxNTkiIHkxPSI1Ni42ODI5IiB4Mj0iMTE4LjQ5MzQiIHkyPSIxLjgyMjUiIGdyYWRpZW50VHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgLTUzLjI5NzQgNjYuNDMyMSkiPiA8c3RvcCBvZmZzZXQ9IjAiIHN0eWxlPSJzdG9wLWNvbG9yOiMzODdFQjgiLz4gPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1jb2xvcjojMzY2OTk0Ii8+PC9saW5lYXJHcmFkaWVudD48cGF0aCBmaWxsPSJ1cmwoI1NWR0lEXzFfKSIgZD0iTTU1LjAyMy0wLjA3N2MtMjUuOTcxLDAtMjYuMjUsMTAuMDgxLTI2LjI1LDEyLjE1NmMwLDMuMTQ4LDAsMTIuNTk0LDAsMTIuNTk0aDI2Ljc1djMuNzgxIGMwLDAtMjcuODUyLDAtMzcuMzc1LDBjLTcuOTQ5LDAtMTcuOTM4LDQuODMzLTE3LjkzOCwyNi4yNWMwLDE5LjY3Myw3Ljc5MiwyNy4yODEsMTUuNjU2LDI3LjI4MWMyLjMzNSwwLDkuMzQ0LDAsOS4zNDQsMCBzMC05Ljc2NSwwLTEzLjEyNWMwLTUuNDkxLDIuNzIxLTE1LjY1NiwxNS40MDYtMTUuNjU2YzE1LjkxLDAsMTkuOTcxLDAsMjYuNTMxLDBjMy45MDIsMCwxNC45MDYtMS42OTYsMTQuOTA2LTE0LjQwNiBjMC0xMy40NTIsMC0xNy44OSwwLTI0LjIxOUM4Mi4wNTQsMTEuNDI2LDgxLjUxNS0wLjA3Nyw1NS4wMjMtMC4wNzd6IE00MC4yNzMsOC4zOTJjMi42NjIsMCw0LjgxMywyLjE1LDQuODEzLDQuODEzIGMwLDIuNjYxLTIuMTUxLDQuODEzLTQuODEzLDQuODEzcy00LjgxMy0yLjE1MS00LjgxMy00LjgxM0MzNS40NiwxMC41NDIsMzcuNjExLDguMzkyLDQwLjI3Myw4LjM5MnoiLz48bGluZWFyR3JhZGllbnQgaWQ9IlNWR0lEXzJfIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeDE9Ijk3LjA0NDQiIHkxPSIyMS42MzIxIiB4Mj0iMTU1LjY2NjUiIHkyPSItMzQuNTMwOCIgZ3JhZGllbnRUcmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAtNTMuMjk3NCA2Ni40MzIxKSI+IDxzdG9wIG9mZnNldD0iMCIgc3R5bGU9InN0b3AtY29sb3I6I0ZGRTA1MiIvPiA8c3RvcCBvZmZzZXQ9IjEiIHN0eWxlPSJzdG9wLWNvbG9yOiNGRkMzMzEiLz48L2xpbmVhckdyYWRpZW50PjxwYXRoIGZpbGw9InVybCgjU1ZHSURfMl8pIiBkPSJNNTUuMzk3LDEwOS45MjNjMjUuOTU5LDAsMjYuMjgyLTEwLjI3MSwyNi4yODItMTIuMTU2YzAtMy4xNDgsMC0xMi41OTQsMC0xMi41OTRINTQuODk3di0zLjc4MSBjMCwwLDI4LjAzMiwwLDM3LjM3NSwwYzguMDA5LDAsMTcuOTM4LTQuOTU0LDE3LjkzOC0yNi4yNWMwLTIzLjMyMi0xMC41MzgtMjcuMjgxLTE1LjY1Ni0yNy4yODFjLTIuMzM2LDAtOS4zNDQsMC05LjM0NCwwIHMwLDEwLjIxNiwwLDEzLjEyNWMwLDUuNDkxLTIuNjMxLDE1LjY1Ni0xNS40MDYsMTUuNjU2Yy0xNS45MSwwLTE5LjQ3NiwwLTI2LjUzMiwwYy0zLjg5MiwwLTE0LjkwNiwxLjg5Ni0xNC45MDYsMTQuNDA2IGMwLDE0LjQ3NSwwLDE4LjI2NSwwLDI0LjIxOUMyOC4zNjYsMTAwLjQ5NywzMS41NjIsMTA5LjkyMyw1NS4zOTcsMTA5LjkyM3ogTTcwLjE0OCwxMDEuNDU0Yy0yLjY2MiwwLTQuODEzLTIuMTUxLTQuODEzLTQuODEzIHMyLjE1LTQuODEzLDQuODEzLTQuODEzYzIuNjYxLDAsNC44MTMsMi4xNTEsNC44MTMsNC44MTNTNzIuODA5LDEwMS40NTQsNzAuMTQ4LDEwMS40NTR6Ii8+PC9zdmc+
)](https://www.python.org)
![Checks](https://github.com/JannisStraus/VPShuntDetector/actions/workflows/checks.yml/badge.svg)
![Coverage](https://codecov.io/gh/JannisStraus/VPShuntDetector/graph/badge.svg?branch=HEAD)

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
`--instructions` flag (see [Instruction Images](#instruction-images) for more
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

In the output directory specified by the `-o` or `--output` option,
VPShuntDetector generates:

- **results.csv:** A CSV file containing, for every input image, the predicted
label and its confidence score for each of the 5 folds. This file also includes
the final aggregated confidence and prediction.
- **Output Images:** For every input image, an output image is saved with a
bounding box overlay that highlights the predicted valve model.
- **Instruction Display:** If an instruction directory is provided via
`--instructions`, the corresponding instruction image for the predicted valve
model will be displayed alongside the output image.
