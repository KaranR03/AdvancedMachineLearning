# Project 8: Extending Tiny NeRF (final pair)

Task: extend the Tutorial 10.4 Tiny NeRF with view-dependent colour (Task 1) and hierarchical
sampling with coarse and fine networks (Task 2), and compare it with Tiny NeRF on the six
held-out views (Task 3).

## What is here

| file | what it is |
|---|---|
| `project8_report.pdf` | the report upload, 2 pages |
| `project8_code.zip` | the code upload, flat, 18 entries |
| `project8_code/` | the same archive unpacked, for reading without unzipping |
| `report_src/` | the notebook patcher, report builder, figure scripts and the checks |

| file | bytes | sha256 |
|---|---|---|
| `project8_report.pdf` | 431,323 | `cdf102d3e95f0a47a2b558b41810b270b0d38bd913a809890d97a8839668d8f2` |
| `project8_code.zip` | 18,986,972 | `c3d8077470bdb013fa6cb1b6a6cbce45a3f410020a0794ea5220cf377c970262` |

## Result

| model | params | evaluations per ray | mean test PSNR (dB) | SSIM |
|---|---|---|---|---|
| Tiny NeRF | 22,148 | 64 | 25.41 | 0.894 |
| Extended NeRF | 96,904 | 128 | 28.44 | 0.940 |
| No view dirs (abl. A) | 96,904 | 128 | 26.41 | 0.917 |
| Uniform 96 (abl. B) | 48,452 | 96 | 28.34 | 0.937 |

- Extended NeRF against Tiny NeRF: +3.03 dB, higher on 6 of 6 views
- view dependence (against ablation A): +2.03 dB, higher on 6 of 6 views
- hierarchical sampling (against ablation B): +0.10 dB, higher on 3 of 6 views

## How it was built

- The trained weights and the two training notebooks are Karan's draft (`Ass8/`), not retrained.
  The only code change to them makes the NeRF docstring a raw string, which removes a
  SyntaxWarning; every code cell's syntax tree is checked identical before its outputs carry over.
- `main_report.ipynb` trains nothing. It loads the four checkpoints, prints every number and draws
  every figure the report carries; `report_src/build_report.py` reads the report's numbers out of
  that printed output.
- Every stored output came from the IFN680 GPU node: device cuda,
  NVIDIA A16-4Q, torch 2.13.0+cu126.
- `report_src/verify.py` checks the names, the 2-page limit, notebook hygiene, the starter's
  provided functions and that every number in the report is printed by `main_report.ipynb`.
  The archive was also run from scratch on the Hub GPU (`hub_graded_run.py`) and on a CPU
  (`clean_room.py`), with no errors and the same printed numbers.

Models and training by Karan Rooprai; final build, evaluation additions, report and checks by
Nhu Hieu Nguyen.
