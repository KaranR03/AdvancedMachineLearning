IFN680 Project 5 - Extending Addition LLM
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  LLMForward.ipynb          trains the left-to-right model on addition and subtraction
  LLMReverse.ipynb          trains the right-to-left model on the same data
  main_report.ipynb         Task 3: loads both models and reproduces every number and
                            figure the report carries. Contains no training loop.
  LLMForward.pth            trained weights, a plain state_dict
  LLMReverse.pth
  project5_testset.pkl      the shared held-out set, 10,000 examples, balanced between
                            addition and subtraction
  LLM*_history.json         validation curves from the two training runs
  LLM*_history_80k.json     validation curves from the same runs on 80,000 examples,
                            which is the data-efficiency comparison in the report
  figure_1_overview.pdf     the two figures the report embeds, as main_report.ipynb
  figure_2_by_length.pdf    writes them

HOW TO RUN
  Unzip everything into one folder and run main_report.ipynb top to bottom. It needs no
  arguments and no data beyond what is in this archive, and takes about a minute on CPU.

  The two training notebooks reproduce the checkpoints from scratch. Thirty epochs on the
  full training set took about seventeen minutes on the IFN680 GPU node, and each notebook
  then repeats the training on a reduced set for the data-efficiency comparison, so expect
  roughly half an hour per notebook in total.

  Every tensor and both models are moved with .to(device), where device is cuda when one is
  visible and cpu otherwise, so the notebooks run unchanged on either. The two training
  notebooks were run on the IFN680 GPU node and main_report.ipynb on a CPU, and the accuracies
  they report agree to the last digit: decoding is greedy, so the device changes the speed and
  not the result.

  The held-out test set is regenerated from SEED inside main_report.ipynb and compared
  against the shipped project5_testset.pkl, so the split can be audited without trusting
  the pickle.
