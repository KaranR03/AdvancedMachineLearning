IFN680 Project 8 - Extending Tiny NeRF
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  TinyNeRF.ipynb              the Tutorial 10.4 Tiny NeRF, trained for 100 epochs and evaluated
                              on the six held-out views (images 100 to 105)
  ExtendedNeRF.ipynb          Task 1 (view-dependent colour) and Task 2 (hierarchical sampling
                              with sample_pdf), plus the two ablations: A, no view direction,
                              and B, view-dependent colour with 96 uniform samples
  main_report.ipynb           Task 3: loads the four checkpoints and reproduces every number and
                              figure in the report. Contains no training loop.
  tinynerf.pth                trained weights, final epoch: Tiny NeRF; Extended NeRF (coarse and
  extended_nerf.pth           fine networks); ablation A (coarse and fine); ablation B (one
  ablation_no_viewdirs.pth    network)
  ablation_uniform96.pth
  training_history.json       every model's train and test PSNR and seconds per epoch
  training_log.txt            the training notebooks' per-epoch log
  tiny_nerf_data.npz          the tutorial's dataset: 106 images at 100 x 100, poses, focal length
  results_summary.json        every number main_report.ipynb prints, as JSON
  figure_*.pdf                the five report figures, written by main_report.ipynb
  nerf_turntable.gif          the 40-frame novel-view turntable, Tiny NeRF | Extended NeRF

HOW TO RUN
  Unzip everything into one folder and run main_report.ipynb top to bottom. It needs no
  arguments and no data beyond what is in this archive, and takes under a minute on the IFN680
  GPU server and about 7 minutes on a CPU. Rendering is deterministic, so it prints the same
  numbers each time; on a different device the last printed digit can differ by float
  rounding, and the render times differ.

  The stored outputs of all three notebooks come from the IFN680 GPU environment. The training
  notebooks retrain every model from scratch (about 3.5 hours on that GPU).
