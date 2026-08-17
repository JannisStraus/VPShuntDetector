# Test fixture attribution

The integration tests download two radiographs at run time from the **CSF Shunt
Valves Dataset**. No image is redistributed with this repository.

- Source: <https://github.com/CSFShuntvalves/xray_csf_shuntvalves>
- Licence: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Pinned commit: `f0937bf54b07a2ba820a5b1c2cfb7c2a86901576`

| Fixture | Path in the dataset |
| ------- | ------------------- |
| `codmanhakim_2.png` | `codman_hakim_programmable/codmanhakim_2.png` |
| `certas_2.png` | `codman_certas_plus/certas_2.png` |

Both are pinned by SHA-256 in `tests/test_all.py` and cached under
`~/.cache/VPShuntDetector/test-images/`, so a run only downloads them once.

## Citation

> Rhomberg, T., Trivik-Barrientos, F., Hakim, A. et al. Applied deep learning in
> neurosurgery: identifying cerebrospinal fluid (CSF) shunt systems in
> hydrocephalus patients. Acta Neurochir 166, 69 (2024).
> <https://doi.org/10.1007/s00701-024-05940-3>

## Why only two images

These are tight crops around the valve, whereas this project's model localises
valves within full skull radiographs. Most images in the source dataset
therefore yield no detection here. These two were selected because they produce
a stable, boxed detection, which is what the integration test needs. They
exercise the pipeline; they are not a benchmark of clinical performance.
