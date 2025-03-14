# MultiScale3D：A MultiScale Fusion Algorithm for Action Recognition
The code is based on Pyskl
![](./img/structure.jpg)

## Dataset
The pkl files of our keypoint datasets are from pyskl. You can also generate keypoint heatmaps with YOLO-Pose and stack them into pkl files, but we suggest first training YOLO-Pose on your video dataset, which requires labeling. While YOLO-Pose matches HRNet in close-range recognition, its generalization drops significantly in long-range cases. However, this isn't a big issue since most figures in public datasets are in close range.

