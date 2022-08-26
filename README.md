## Adversarial Detection

> Attacking End-to-End Object Detection Systems

### Overview

Generating adversarial patch is as easy as **drag and drop**.

![](doc/attack.png)

### Quick Start

#### Step 1: Setup the server

You may use [anaconda](https://www.continuum.io/downloads) or [miniconda](https://conda.io/miniconda.html). 

```
$ cd model
$ conda env create -f environment.yml
$ conda activate adversarial-detection
$ python detect.py --model yolov3.h5 --class_name coco_classes.txt
```

#### Step 2: Setup the browser

This is just a website, your can use any web server, just serve all the content under **client/web**.

The client is built as a single executable file.

```
$ client.exe
```

For Linux and Mac, or other Unix, the server can be built with:

```
$ go get -u github.com/gobuffalo/packr/packr
$ go get github.com/gobuffalo/packr@v1.30.1
$ packr build
```

The web page will be available at: http://localhost:3333/

That's it!
