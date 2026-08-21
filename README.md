# IFN680 — Group 4

Karan Rooprai (n12498122) · Nhu Hieu Nguyen (n12194778)

Shared repository for IFN680 coursework. Projects 1–2 were individual; **projects 3–8 are group
work** and are built here.

## How we work in this repo

**`main` holds submitted work only. Nobody commits to it directly.**

| | |
|---|---|
| Branch naming | `<name>/<project>-<topic>` — `hieu/p3-submission`, `karan/p4-baseline` |
| Merging | open a PR into `main` once a deliverable is final |
| Rewriting history | **don't.** A force-push to `main` on 21 Aug silently deleted the Project 2 marker feedback; it survived only because a copy had been taken. `git push --force` on a shared branch destroys work that is not yours |

## Layout

```
Ass1/   Project 1 — polynomial regression        (individual, 4.5/5)
Ass2/   Project 2 — chest X-ray diagnosis        (individual, 4.5/5)
Ass3/   Project 3 — aircraft classification      (group)
  development.ipynb     trains baseline, three experiments, final model
  main_report.ipynb     reproduces every figure and metric; retrains nothing
  best_model.pth        final weights, refit on the whole trainval split
  histories.pkl         learning curves for every run
  project3_report.pdf   the 2-page report as submitted
  report_src/           LaTeX source and the build/verify scripts behind it
```

## What never goes in this repo

- **The image datasets.** `FGVCAircraft_Subset20.zip` is 158 MB and is supplied with the task.
- **Iterative checkpoints.** Git keeps every version of a binary forever, so five saves of a 45 MB
  `.pth` is 225 MB of permanent history. Commit the *one* checkpoint that ships with a submission,
  at submission time, and nothing else. GitHub warns past 50 MB and refuses past 100 MB.

## Marker feedback

Per-project feedback in `feedback/`. It does not appear anywhere in Canvas, so if it is not kept
here it is not kept anywhere.

The rule it has cost us half a mark on twice, quoted from the Project 2 remarks:

> "The confusion matrix and all results should be reported on the testing set after training on
> train plus validation data."

Both Project 1 and Project 2 lost 0.5 on the report to this. Project 3 satisfies it: the final
model is retrained on the entire `trainval` split and scored once on the test set.
