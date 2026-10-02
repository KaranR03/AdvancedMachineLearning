# Project 7: Improving DDPM (final pair)

Hypothesis tested: a cosine schedule allows a denoising diffusion probabilistic model to generate
high-quality samples using fewer time steps than a linear schedule.

## What is here

| file | what it is |
|---|---|
| `project7_report.pdf` | the report upload, 2 pages |
| `project7_code.zip` | the code upload, flat, 13 entries |
| `project7_code/` | the same archive unpacked, for reading without unzipping |
| `report_src/` | the notebook generator, report builder, figure script and the checks |

| file | bytes | sha256 |
|---|---|---|
| `project7_report.pdf` | 152,991 | `4e8b3c4cfdb4d58631abecc1e958665ddaf690cf6eddaa80d749bf5462148262` |
| `project7_code.zip` | 17,937,954 | `25a71d9935134788e791c6b38a10b01472b920de5cbd67ef6d973e2e4a22a805` |

## Result

- Frechet distance at the full 1000 steps (real-against-real floor 7.65): linear
  17.12, cosine 13.5
- class consistency at 1000 steps on 1,000 samples: linear
  0.9650, cosine 0.9640 (the 12x10 grids alone read
  0.9583 and 0.925); the classifier
  scores 0.9914 on real test digits
- ddpm: cosine better at K = 10, 20 | linear better at K = none | unclear at K = 50, 100, 250
- ddim: cosine better at K = none | linear better at K = none | unclear at K = 10, 20, 50, 100, 250

The cosine schedule counts as better at a step count K when its Frechet distance is lower, the
paired bootstrap interval on the difference lies below 0, and the sign is the same at all three
training seeds. The rule was written before any sample was scored.

## How it was built

- One generator (`report_src/build_notebooks.py`) writes all three notebooks, so the network,
  schedule and sampler cells are byte-identical wherever they appear, and the tutorial's cells are
  read from the Week 9 solution rather than retyped.
- Both schedules trained the tutorial's `UNet_cond` for 150 epochs at batch
  128, seeds [0, 1, 2]; only the beta schedule differs between them.
- `main_report.ipynb` trains nothing. It loads the checkpoints, prints every number and draws every
  figure the report carries; `report_src/build_report.py` reads the report's numbers out of that
  printed output.
- Every stored output came from the IFN680 GPU node: device cuda:0,
  torch 2.13.0+cu126, python 3.12.14.
- `report_src/verify.py` checks the names, the 2-page limit, notebook hygiene, the tutorial's cells
  and that every number in the report is printed by `main_report.ipynb`. `clean_room.py` runs
  `main_report.ipynb` from the archive alone.

Built and run by Nhu Hieu Nguyen.
