## Adversarial Detection

> Attacking End-to-End Object Detection Systems

### Overview

Generating adversarial patch is as easy as **drag and drop**.

![](doc/attack.png)

### Quick Start

You may use [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html). 

```
$ git clone https://github.com/wuhanstudio/adversarial-detection
$ cd adversarial-detection/model

$ # CPU
$ conda env create -f environment.yml
$ conda activate adversarial-detection

$ # GPU
$ conda env create -f environment_gpu.yml
$ conda activate adversarial-gpu-detection

$ python detect.py --model yolov3-tiny.h5 --class_name coco_classes.txt
```

The web page will be available at: http://localhost:9090/

That's it!
