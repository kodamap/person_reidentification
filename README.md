<!-- TOC -->

- [Person Re-identification with OpenVINO](#person-re-identification-with-openvino)
    - [What's this](#whats-this)
    - [Reference](#reference)
        - [OpenVINO Toolkit and Flask Video streaming](#openvino-toolkit-and-flask-video-streaming)
        - [OpenVINO Intel Model](#openvino-intel-model)
    - [Tested Environment](#tested-environment)
    - [Required Python packages](#required-python-packages)
    - [How to use](#how-to-use)
    - [Run app](#run-app)

<!-- /TOC -->

# Person Re-identification with OpenVINO

## What's this

This is Person Identification Test App using Intel OpenVINO Person Re-Identification Model.

You can do followings:

* Person Detection
* Preson Re-Identification (Tracking and Counter)

**Person re-identifiction - Tracking - (YouTube Link)**
<a href="https://youtu.be/zIkzlB-Z-vU">
<img src="https://raw.githubusercontent.com/wiki/kodamap/person_reidentification/images/mall.gif" alt="mall" width="%" height="auto"></a>


**Person re-identifiction - Tracking and Counter- (YouTube Link)**
<a href="https://youtu.be/Pj6HYWWyucU">
<img src="https://raw.githubusercontent.com/wiki/kodamap/person_reidentification/images/person_reid.gif" alt="person reid" width="%" height="auto"></a>

## Reference

### OpenVINO Toolkit and Flask Video streaming

* [Install OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html)
* [Flask Video streaming](https://github.com/miguelgrinberg/flask-video-streaming)

### OpenVINO Intel Model

* [person-detection-retail-0013](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/person-detection-retail-0013/description/person-detection-retail-0013.md)
* [person-reidentification-retail-0031](https://github.com/openvinotoolkit/open_model_zoo/blob/2020.3/models/intel/person-reidentification-retail-0031/description/person-reidentification-retail-0031.md)



## Tested Environment

- Python 3.7.6 (need 3.6+ for f-strings)
- Windows 10 [Version 10.0.19041.388]
- OpenVINO Toolkit 2020.1+[^1]

[^1]: openvino.inference_engine version openvino_2020.1.033 or above build does not need cpu extension.
    # https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit/topic/848825


## Required Python packages

```sh
pip install -r requirements.txt
```

## How to use

```sh
python app.py -h
usage: app.py [-h] -i INPUT [-d {CPU,GPU,FPGA,MYRIAD}]
              [-d_reid {CPU,GPU,FPGA,MYRIAD}] [--v4l] [-ax {0,1}] [-g GRID]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to video file or image. 'cam' for capturing video
                        stream from camera
  -d {CPU,GPU,FPGA,MYRIAD}, --device {CPU,GPU,FPGA,MYRIAD}
                        Specify the target device for Person Detection to
                        infer on; CPU, GPU, FPGA or MYRIAD is acceptable.
  -d_reid {CPU,GPU,FPGA,MYRIAD}, --device_reidentification {CPU,GPU,FPGA,MYRIAD}
                        Specify the target device for person re-identificaiton
                        to infer on; CPU, GPU, FPGA or MYRIAD is acceptable.
  --v4l                 cv2.VideoCapture with cv2.CAP_V4L
  -ax {0,1}, --axis {0,1}
                        Specify the axis when counting person horizontally or
                        vertically (0: count horizontally(x-axis) , 1: count
                        vertically (y-axis)
  -g GRID, --grid GRID  Specify how many grid to divide frame. This is used to
                        define boundary area when the tracker counts person. 0
                        ~ 2 means not counting person. (range: 3 < max_grid)

```


## Run app

**example1.** camera streaming without person counter 

```sh
python app.py -i cam
```

**example2** Specify video file with person counter

```py
python app.py -i P1033674_1920.mp4 --grid 10 
```


Access the url bellow on your browser

```txt
http://127.0.0.1:5000/
```
