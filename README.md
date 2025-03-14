# MultiScale3D：A MultiScale Fusion Algorithm for Action Recognition
The code is based on Pyskl
![](./img/structure.jpg)

## Project Structure

```
── config
│   ├── Posec3d
├── core
│   ├── loss
│   │   ├── AutomaticWeightedLoss.py
│   │   ├── SupervisedContrastiveLoss.py
│   │   └── UnsupervisedContrastiveLoss.py
│   ├── model
│   │   ├── backbone
│   │   │   ├── FC.py
│   │   │   ├── sage.py
│   │   ├── Classifier.py
│   │   ├── Encoder.py
│   │   ├── MainModel.py
│   │   └── Voter.py
│   ├── multimodal_dataset.py
│   ├── aug.py
│   ├── ita.py
│   └── TVDiag.py
├── data
│   ├── gaia
│   │   ├── label.csv
│   │   ├── raw
│   │   └── tmp
│   └── sockshop
│       ├── label.csv
│       ├── raw
│       └── tmp
├── helper
│   ├── complexity.py
│   ├── early_stop.py
│   ├── eval.py
│   ├── io_util.py
│   ├── logger.py
│   ├── Result.py
│   ├── scaler.py
│   ├── seed.py
│   └── time_util.py
├── process
│   ├── EventProcess.py
│   └── events
│       ├── cbow.py
│       ├── cnn1d_w2v.py
│       ├── fasttext_w2v.py
│       └── lda_w2v.py
├── LICENSE
├── main.py
├── README.md
```
