IFN680 Project 4 - AI Auditing
Group 4: Karan Rooprai (n12498122), Nhu Hieu Nguyen (n12194778)

CONTENTS
  case1.ipynb   clean-shaven detector, studio to mobile
  case2.ipynb   eyeglasses detector for a retail kiosk
  case3.ipynb   'appears young' tagger with a high-confidence bypass
  caseN_predictions.pkl
                the labels and positive-class probabilities each notebook computes, cached
                so the metrics can be re-derived without a second pass over the images

HOW TO RUN
  Unzip the three supplied datasets and copy the three supplied checkpoints beside these
  notebooks, so that the layout is:

      case1.ipynb  case2.ipynb  case3.ipynb
      Case1/Case1Dataset/test_internal/{negative,positive}
      Case1/Case1Dataset/test_external/{negative,positive}
      Case1/resnet_frozen_best.pth
      Case2/...   Case3/...

  Then run each notebook top to bottom. The images and weights are supplied with the
  assessment and are not duplicated here.

  Each notebook also accepts CaseNDataset/ directly beside it, or the same layout under
  ~/Assessment4/. It searches those locations in order and stops with a message naming the
  expected layout if none of them holds the data.

  All three checkpoints are named resnet_frozen_best.pth but they are three different
  models. Each notebook checks the MD5 of the file it loaded against the one for its own
  case, so putting all three in one folder fails immediately instead of producing wrong
  numbers quietly.

  Every tensor and the model are moved with .to(device), where device is cuda:0 when one
  is visible and cpu otherwise, so the notebooks run unchanged on either. The stored
  outputs are from a CPU run. The same notebooks were also run on the IFN680 GPU node
  (torch 2.13.0+cu126, device cuda:0): of the 83 lines they print, the only differences
  are the four that name the environment and four values that move in the third decimal,
  where a borderline probability lands on the other side of 0.5 under a different kernel.
  Every figure quoted in the report is identical on both devices.
