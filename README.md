# Camera-based Hand Tracking using a Mirror-based Multi-view Setup

![](video/hand_only.gif)

This repo contains the sample code for the paper: Camera-based Hand Tracking using a Mirror-based Multi-view Setup.

[**Paper**](https://doi.org/10.1109/EMBC44109.2020.9176728) | [**Project**](https://gmntu.github.io/mirror/)


If you find our work useful, please consider citing
```BibTeX
@inproceedings{mirror:2020,
  title     = {Camera-based Hand Tracking using a Mirror-based Multi-view Setup},
  author    = {Guan Ming, Lim and Prayook, Jatesiktat and Christopher Wee Keong, Kuah and Wei Tech, Ang},
  booktitle = {42st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  year      = {2020}
}
```

## Usage
If the camera intrinsics and extrinsics have been calibrated and saved, can proceed to step 4) directly

1) Calibrate camera intrinsics from video images of chessboard pattern placed at different pose
```
python 00_calib_intrin_video.py
```

2) Prepare data for calibrating camera extrinsics. Note: This program requires manual input to guide the process of whether to flip the projected axis (for mirror image) and number the camera views (in clockwise order starting from the actual camera view)
```
python 01_prep_extrin_video.py
```

3) Calibrate camera extrinsics. Note: This program uses a [bundle-adjustment like method](https://ieeexplore.ieee.org/document/6696517) and the optimization process uses [lmfit](https://lmfit.github.io/lmfit-py/)
```
python 02_calib_extrin_video.py 
```

4) Perform model fitting loop. Note: This program assumes the bounding boxes (ROI) of the hand have been determined in the first frame and then perform the subsequent image processing pipelines (2D hand keypoint detection -> 3D model fitting) automatically. This offline processing will save images of 2D hand keypoint and projected 3D hand skeleton, as well as generate a video.mp4 file to view the qualitative result.
```
python 07_model_fitting_video.py --file ball
```

## Qualitative Results

1) Hand interacting with a ball
![](video/hand_with_ball.gif)

2) Hand interacting with a cup
![](video/hand_with_cup.gif)

3) Hand interacting with a cube (length of 5 cm)
![](video/hand_with_cube_small.gif)

4) Hand interacting with a bigger cube (length of 7.5 cm)
![](video/hand_with_cube_big.gif)