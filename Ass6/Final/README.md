# Project 6: The Sharpness Quest (final pair)

Hypothesis tested: adding a discriminator term to the loss function of a cVAE during training
increases the sharpness of the reconstructed images.

## What is here

| file | what it is |
|---|---|
| `project6_report.pdf` | the report upload, 2 pages |
| `project6_code.zip` | the code upload, flat, 13 entries |
| `project6_code/` | the same archive unpacked, for reading without unzipping |
| `report_src/` | the notebook generator, report builder, figure script and the checks |

| file | bytes | sha256 |
|---|---|---|
| `project6_report.pdf` | 147,890 | `fbf46996e734a419a806292d81d4d42054a1983926a63f6a59d0159c43a04cf7` |
| `project6_code.zip` | 14,373,186 | `c7e5d6ac34d8b55537592fd11820e7aa53e185612467ed144d1d293d1827bb4f` |

## Result

- lambda, chosen on a development split before the test set was touched: **0.3**
- reconstructions with the discriminator loss: **sharper by the fixed rule**
- samples with the discriminator loss: **sharper by the fixed rule**
- classifier accuracy on real test digits 0.9836; on samples,
  baseline 0.9855 and discriminator loss
  0.9875
- Frechet distance of samples (floor 0.45): baseline
  51.32, discriminator loss
  4.81

"Sharper" means a higher Laplacian variance whose paired bootstrap interval lies above 0, the same
sign at all three training seeds, and a lower mid-grey fraction agreeing with it. The rule was
written before any test number existed.

## How it was built

- One generator (`report_src/build_notebooks.py`) writes all four notebooks, so the model, loss
  and data cells are byte-identical wherever they appear.
- Both cVAEs trained 150 epochs on a cosine learning-rate schedule, seeds
  [0, 1, 2]; lambda and convergence were settled on a random 10,000 development split,
  then the final models were retrained on all 60,000 training images.
- `main_report.ipynb` trains nothing. It loads the three checkpoints and prints every number and
  draws both figures the report carries; `report_src/build_report.py` reads the report's numbers
  out of that printed output.
- Every stored output came from the IFN680 GPU node: device cuda:0,
  torch 2.13.0+cu126, python 3.12.14.
- `report_src/verify.py` checks the names, the 2-page limit, notebook hygiene, the data protocol
  and that every number in the report is printed by `main_report.ipynb`. `clean_room.py` runs
  `main_report.ipynb` from the archive alone.

Built and run by Nhu Hieu Nguyen.
