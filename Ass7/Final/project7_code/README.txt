IFN680 Project 7 - Improving DDPM
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  DDPM_CosineSchedule.ipynb   Task 1: trains the conditional DDPM of Tutorial 9.3 twice, with the
                              tutorial's linear_beta_schedule and with a cosine_beta_schedule,
                              three training seeds each, and defines the samplers
  DigitClassifier.ipynb       trains the standalone digit classifier used for evaluation
  main_report.ipynb           Task 2: loads the checkpoints and reproduces every number and figure
                              in the report, including the 12x10 grid of generated digits for
                              both schedules. Contains no training loop.
  ddpm_linear.pth             trained U-Nets, plain state_dicts: the final model of each training
  ddpm_cosine.pth             seed (seed_0, seed_1, seed_2) and seed 0's earlier snapshots
                              (epoch_10, epoch_25, epoch_50, epoch_100)
  DigitClassifier.pth         the classifier's weights
  DDPM_history.json           every run's loss and seconds per epoch, and the training settings
  mnist_custom_test.pt        the 10,000-image test split of mnist_custom.pt, the only data
                              main_report.ipynb reads
  figure_*.pdf                the four figures main_report.ipynb writes; figures 1 and 2 are the
                              report's

HOW TO RUN
  Unzip everything into one folder and run main_report.ipynb top to bottom. It needs no
  arguments and no data beyond what is in this archive. It samples tens of thousands of digits,
  which takes about 9 minutes on the IFN680 GPU server and about 105 minutes on a CPU.
  Every random draw is seeded, so it prints the same numbers each time; on a different device
  the last printed digit can differ by float rounding.

  The stored outputs of all three notebooks come from the IFN680 GPU environment. The training
  notebooks need mnist_custom.pt, downloaded from this project's Canvas page into the same folder.
