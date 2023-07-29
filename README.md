## Adversarial Detection

> Attacking Object Detection Systems in Real Time

[[ Talk ]](https://detection.wuhanstudio.uk) [[ Video ]](https://youtu.be/zJZ1aNlXsMU) [[ Code ]](https://github.com/wuhanstudio/adversarial-detection) [[ Paper ]](https://arxiv.org/abs/2209.01962) 

### Overview

Generating adversarial patch is as easy as **drag and drop**.

![](doc/attack.png)

### Quick Start

You may use [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html). 

```
$ git clone https://github.com/wuhanstudio/adversarial-detection
$ cd adversarial-detection

$ # CPU
$ conda env create -f environment.yml
$ conda activate adversarial-detection

$ # GPU
$ conda env create -f environment_gpu.yml
$ conda activate adversarial-gpu-detection

# Pre-trained models are available here
# https://github.com/wuhanstudio/adversarial-detection/releases

$ python detect.py --model model/yolov3-tiny.h5 --class_name coco_classes.txt
```

The web page will be available at: http://localhost:9090/

That's it!

## Adversarial ROS Detection

We also tested our attacks in ROS Gazebo simulator. 

https://github.com/wuhanstudio/adversarial-ros-detection

[![](https://raw.githubusercontent.com/wuhanstudio/adversarial-ros-detection/master/doc/demo.jpg)](https://github.com/wuhanstudio/adversarial-ros-detection)

## Citation

```
@INPROCEEDINGS{han2023detection,
  author={Wu, Han and Yunas, Syed and Rowlands, Sareh and Ruan, Wenjie and Wahlström, Johan},
  booktitle={2023 IEEE Intelligent Vehicles Symposium (IV)}, 
  title={Adversarial Detection: Attacking Object Detection in Real Time}, 
  year={2023},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/IV55152.2023.10186608}
}
```
