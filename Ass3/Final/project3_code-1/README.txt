IFN680 Project 3 - Aircraft Classification
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  development.ipynb   trains the baseline, the three experiments and the final model,
                      and writes best_model.pth and histories.pkl
  main_report.ipynb   reproduces every figure and metric in the report. It retrains
                      nothing: it reloads the two files above.
  best_model.pth      weights of the final model, refit on the whole trainval split
  histories.pkl       learning curves and validation scores for all runs

HOW TO RUN
  Unzip FGVCAircraft_Subset20.zip into this same folder, so that the layout is:

      main_report.ipynb
      FGVCAircraft_Subset20/trainval/class_00 ... class_19
      FGVCAircraft_Subset20/test/class_00 ... class_19

  Then run main_report.ipynb top to bottom. It checks for the dataset in its first cell
  and stops with a readable message if it is somewhere else.

  The image set is not included here because it is 158 MB and is supplied with the task.
