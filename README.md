# Steel Defects Classification and Segmentation
=======================================================================

## A metal sheet defect classifier, locater using PyTorch

You can view the 1st part of the project here [github.com/mahekvora21/Metal-Sheel-Defects-Binary-Classification](https://github.com/mahekvora21/Metal-Sheel-Defects-Binary-Classification)

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── datasets           <- File containig the dataloaders
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── config             
    │   └── cls            <- Configuration files for classification models
    │   └── seg            <- Configuration files for segmentation models
    │
    ├── utils              <- Here functions, metrics, callbacks are defined
    │
    ├── losses             <- Classes for losses are defined
    │
    ├── optimizers         <- Classes for optimizers are defined
    │
    ├── schedulers         <- Classes for schedulers are defined
    │
    ├── transforms 
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, PPT etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
## Instructions to run the models
Environment setup
------------
1. Install modules from requirements.txt
2. If you are using google colab run the following with gpu enabled
```
from google.colab import drive
drive.mount("/content/drive")

#to confirm gpu access
import tensorflow as tf
device_name=tf.test.gpu_device_name()
print(device_name)
```
Classification models
------------
```
python split_folds.py --config config/base_config.yml
python train_cls.py --config config/cls/001_resnet50_BCE_5class_fold0.yml
python train_cls.py --config config/cls/002_efnet_b3_cls_BCE_5class_fold1.yml
python train_cls.py --config config/cls/003_seresnext50_cls_BCE_5class_fold2.yml
```

Segmentation models
------------
```
python train_seg.py --config config/seg/001_resnet18_Unet_fold0.yml
python train_seg.py --config config/seg/002_resnet18_PSPNet_fold0.yml
python train_seg.py --config config/seg/003_resnet50_fpn_fold0.yml
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
