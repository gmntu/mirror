[Camera-based Hand Tracking using a Mirror-based Multi-view Setup](https://www.youtube.com/embed/X8sVhl8Wswk)
<img src="video/hand_only.gif">

This repository contains sample codes to demonstrate the potential of **dynamic** hand pose estimation using a **multi-view** setup with a readily available **color camera** and **plane mirrors**.

All the videos and images are captured on a mobile phone and post-processed on a computer to generate 2D hand keypoints with bounding boxes of the hand (left column) and 3D hand skeleton projected onto individual view (right column).

If you find our code or paper useful, please consider citing
```
@inproceedings{mirror:2020,
  title = {Camera-based Hand Tracking using a Mirror-based Multi-view Setup},
  author = {Guan Ming, Lim and Prayook, Jatesiktat and Christopher Wee Keong, Kuah and Wei Tech, Ang},
  booktitle = {42st Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  year = {2020}
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

1) [Video](https://www.youtube.com/embed/wcudUoM_ZcQ) on hand interacting with a ball
<img src="video/hand_with_ball.gif">

2) [Video](https://www.youtube.com/embed/37z8yIOd7GM) on hand interacting with a cup
<img src="video/hand_with_cup.gif">

3) [Video](https://www.youtube.com/embed/VhW-38FZN6Y) on hand interacting with a cube (length of 5 cm)
<img src="video/hand_with_cube_small.gif">

4) [Video](https://www.youtube.com/embed/QxNZqGyWXUo) on hand interacting with a bigger cube (length of 7.5 cm)
<img src="video/hand_with_cube_big.gif">

