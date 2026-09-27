Project 7 - Improving DDPM (cosine vs linear noise schedule) - Group 4
Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  DDPM_CosineSchedule.ipynb  Task 1: trains two conditional DDPMs on mnist_custom.pt following
                             Tutorial 9.3 - one with the tutorial linear beta schedule and one with our
                             cosine beta schedule (Nichol and Dhariwal, 2021). Saves both checkpoints.
                             Its sanity grids use the tutorial's full 1000-step ancestral sampler.
  main_report.ipynb          Task 2: loads both checkpoints and reproduces every figure and number in the
                             report: the 12 x 10 (120-image) conditional grids for both schedules, and
                             sample quality versus number of reverse steps (DDIM sampler) measured by
                             classifier recognisability and a Frechet feature distance. No DDPM training.
  ddpm_linear.pth, ddpm_cosine.pth   trained eps-model checkpoints (identical U-Net, 100 epochs each)
  classifier.pth             cached evaluation CNN (retrained in about 20 s if missing)
  ddpm_losses.pkl            training loss curves and wall-clock training time per schedule
  results_summary.pkl, a7_results.json   all reported numbers
  fig_quality_vs_steps.png, fig_fewstep_grids.png   report figures
  mnist_custom.pt            the provided dataset (must sit next to the notebooks)

HOW TO RUN (IFN680 GPU environment)
  Keep all files in one folder and run main_report.ipynb top to bottom (about 2 minutes on the GPU).
  All tensors use .to(device); the notebook also runs on CPU, only slower.

HEADLINE RESULT
  Frechet feature distance (lower is better): cosine at 20 steps = 17.8, linear at 1000 steps = 19.0.
  The cosine model reaches better distributional quality with 50x fewer sampling steps; training time
  is identical (about 801 s each). At an extreme 10 steps the linear model is better (22.4 vs 27.9).
