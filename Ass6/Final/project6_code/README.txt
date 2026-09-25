IFN680 Project 6 - The Sharpness Quest
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  cVAE_Baseline.ipynb            Task 1: trains the standard cVAE of Tutorial 8.3, 7 latent
                                 dimensions, adapted to the 20x20 images of mnist_custom.pt
  cVAE_DiscriminatorLoss.ipynb   Task 2: the same cVAE trained with an added discriminator loss,
                                 including the choice of the loss weight lambda
  DigitClassifier.ipynb          trains the standalone digit classifier used for evaluation
  main_report.ipynb              Task 3: loads the three checkpoints and reproduces every number
                                 and figure in the report, including the 12x10 grid of generated
                                 digits for both models. Contains no training loop.
  *.pth                          trained weights, plain state_dicts: one cVAE per training seed
                                 (0, 1, 2) in each cVAE file, plus each seed's discriminator
  *_history.json                 per-epoch training curves, the lambda sweep and the settings
  mnist_custom_test.pt           the 10,000-image test split of mnist_custom.pt, the only data
                                 main_report.ipynb reads
  figure_1_grids.pdf             the report's figures, as main_report.ipynb writes them
  figure_2_reconstructions.pdf

HOW TO RUN
  Unzip everything into one folder and run main_report.ipynb top to bottom. It needs no
  arguments and no data beyond what is in this archive, runs on a GPU or a CPU, and takes a
  few minutes on a CPU. Every random draw is seeded, so it prints the same numbers each time;
  on a different device the last printed digit can differ by float rounding.

  The stored outputs of all four notebooks come from the IFN680 GPU environment. The training
  notebooks need mnist_custom.pt, downloaded from this project's Canvas page into the same folder.
