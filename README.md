# MultiScale3D：A MultiScale Fusion Algorithm for Action Recognition
The code is based on Pyskl
![](./img/structure.jpg)

## Dataset
The pkl files of our keypoint datasets are from pyskl. You can also generate keypoint heatmaps with YOLO-Pose and stack them into pkl files, but we suggest first training YOLO-Pose on your video dataset, which requires labeling. While YOLO-Pose matches HRNet in close-range recognition, its generalization drops significantly in long-range cases. However, this isn't a big issue since most figures in public datasets are in close range.
pkl((https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md))

## Getting Started

To perform keypoint heatmap stacking with YOLO-Pose and convert the result to a pickle file, run the following command after adjusting the paths for the weight file, config file, video_list file, and output directory in the script.
```python
python Yolov8_2D_Skeleton.py
```
The training and testing commands are as follows:
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```
For the NTU dataset, the command to merge the predicted results and get the (J+L) prediction accuracy is as follows:
```python
python ensemble.py
```
