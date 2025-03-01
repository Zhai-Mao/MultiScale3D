"""
生成最终的pkl文件
输入：yolov8-pose获取的关键点
输入文件：pyskl-main\tools\data\Yolov8_2d_skeleton.py
输出：最终pkl文件
"""
from mmcv import load, dump
train = load('E:/pyskl-main/tools/data/Weizmann/train.json')
test = load('E:/pyskl-main/tools/data/Weizmann/test.json')
annotations = load('E:/pyskl-main/tools/data/Weizmann/video.pkl')
split = dict()
split['train'] = [x['vid_name'] for x in train]
split['test'] = [x['vid_name'] for x in test]
dump(dict(split=split, annotations=annotations), 'E:/pyskl-main/tools/data/Weizmann/video_final.pkl')