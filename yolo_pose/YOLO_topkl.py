# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy as cp
import decord
import mmcv
import numpy as np
import os
import os.path as osp
import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from tqdm import tqdm
from ultralytics import YOLO

import pyskl  # noqa: F401
from pyskl.smp import mrlines

try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose  # noqa: F401
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

pyskl_root = osp.dirname(pyskl.__path__[0])
default_config = "yolov8s-pose.yaml"
default_ckpt = "E:/pyskl-main/tools/yolov8n-pose.pt"


def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def detection_inference(frames):
    results = []
    for frame in frames:
        # box = []
        # model = YOLO('yolov8l-pose.yaml')
        model = YOLO(r'E:\pyskl-main\tools\yolov8n-pose.pt')
        # result = inference_detector(model, frame)
        result = model(frame)
        boxes = result[0].boxes
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        # box.append(xyxy)
        results.append(xyxy)
    return results


def pose_inference(anno_in, frames, det_results, compress=False):
    anno = cp.deepcopy(anno_in)
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    anno['total_frames'] = total_frames
    anno['num_person_raw'] = num_person

    if compress:
        kp, frame_inds = [], []
        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(YOLO, f, d, format='xyxy')[0]
            for j, item in enumerate(pose):
                kp.append(item['keypoints'])
                frame_inds.append(i)
        anno['keypoint'] = np.stack(kp).astype(np.float16)
        anno['frame_inds'] = np.array(frame_inds, dtype=np.int16)
    else:
        kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)
        for i, (f, d) in enumerate(zip(frames, det_results)):
            # Align input format
            pose = []
            it = {}

            # d = [dict(bbox=x) for x in list(d)]
            # pose = inference_top_down_pose_model(YOLO(r'E:\pyskl-main\tools\yolov8n-pose.pt'), f, d, format='xyxy')[0]
            model = YOLO(r'E:\pyskl-main\tools\yolov8n-pose.pt')
            # result = inference_detector(model, frame)
            result = model(f)
            boxes = result[0].boxes
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            it['bbox'] = xyxy
            keypoint = result[0].keypoints.data.cpu().numpy()
            keypoint = keypoint[0]
            it['keypoints'] = keypoint
            pose = [it]

            for j, item in enumerate(pose):
                if len(item['keypoints']) == 0:
                    item['keypoints'] = np.zeros((17, 3)).astype(np.float32)
                kp[j, i] = item['keypoints']
        anno['keypoint'] = kp[..., :2].astype(np.float16)
        anno['keypoint_score'] = kp[..., 2].astype(np.float16)
    return anno


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate 2D pose annotations for a custom video dataset')
    # * Both mmdet and mmpose should be installed from source
    # parser.add_argument('--mmdet-root', type=str, default=default_mmdet_root)
    # parser.add_argument('--mmpose-root', type=str, default=default_mmpose_root)
    parser.add_argument('--default-config', type=str, default=default_config)
    parser.add_argument('--default-ckpt', type=str, default=default_ckpt)
    # parser.add_argument('--pose-config', type=str, default=default_pose_config)
    # parser.add_argument('--pose-ckpt', type=str, default=default_pose_ckpt)
    # * Only det boxes with score larger than det_score_thr will be kept
    parser.add_argument('--det-score-thr', type=float, default=0.7)
    # * Only det boxes with large enough sizes will be kept,
    parser.add_argument('--det-area-thr', type=float, default=1600)
    # * Accepted formats for each line in video_list are:
    # * 1. "xxx.mp4" ('label' is missing, the dataset can be used for inference, but not training)
    # * 2. "xxx.mp4 label" ('label' is an integer (category index),
    # * the result can be used for both training & testing)
    # * All lines should take the same format.
    parser.add_argument('--video-list', type=str, default='E:/pyskl-main/tools/data/Weizmann/Weizmann.list', help='the list of source videos')
    # * out should ends with '.pkl'
    parser.add_argument('--out', type=str, default='E:/pyskl-main/tools/data/Weizmann/video.pkl', help='output pickle name')
    parser.add_argument('--tmpdir', type=str, default='tmp')
    parser.add_argument('--local_rank', type=int, default=0)
    # * When non-dist is set, will only use 1 GPU
    parser.add_argument('--non-dist', action='store_true', help='whether to use distributed skeleton extraction')
    parser.add_argument('--compress', action='store_true', help='whether to do K400-style compression')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.out.endswith('.pkl')

    lines1 = mrlines(args.video_list)
    lines = [x.split() for x in lines1]

    # * We set 'frame_dir' as the base name (w/o. suffix) of each video
    assert len(lines[0]) in [1, 2]
    if len(lines[0]) == 1:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0]) for x in lines]
    else:
        annos = [dict(frame_dir=osp.basename(x[0]).split('.')[0], filename=x[0], label=int(x[1])) for x in lines]
    rank = 0  # 添加该
    world_size = 1  # 添加
    # if args.non_dist:
    #     my_part = annos
    #     os.makedirs(args.tmpdir, exist_ok=True)
    # else:
    #     init_dist('pytorch', backend='nccl')
    #     rank, world_size = get_dist_info()
    #     if rank == 0:
    #         os.makedirs(args.tmpdir, exist_ok=True)
    #     dist.barrier()
    #     my_part = annos[rank::world_size]
    my_part = annos

    # det_model = init_detector(args.det_config, args.det_ckpt, 'cuda')
    # assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
    # pose_model = init_pose_model(args.pose_config, args.pose_ckpt, 'cuda')

    results = []
    for anno in tqdm(my_part):
        frames = extract_frame(anno['filename'])
        # det_results1 = detection_inference(det_model, frames)
        det_results = detection_inference(frames)
        # * Get detection results for human
        # det_results = [x[0] for x in det_results1]
        # for i, res in enumerate(det_results):
        #     # * filter boxes with small scores
        #     res = res[res[:, 4] >= args.det_score_thr]
        #     # * filter boxes with small areas
        #     box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
        #     assert np.all(box_areas >= 0)
        #     res = res[box_areas >= args.det_area_thr]
        #     det_results[i] = res

        shape = frames[0].shape[:2]
        anno['img_shape'] = shape
        anno = pose_inference(anno, frames, det_results, compress=args.compress)
        anno.pop('filename')
        results.append(anno)

    if args.non_dist:
        mmcv.dump(results, args.out)
    else:
        mmcv.dump(results, osp.join(args.tmpdir, f'part_{rank}.pkl'))
        dist.barrier()

        if rank == 0:
            parts = [mmcv.load(osp.join(args.tmpdir, f'part_{i}.pkl')) for i in range(world_size)]
            rem = len(annos) % world_size
            if rem:
                for i in range(rem, world_size):
                    parts[i].append(None)

            ordered_results = []
            for res in zip(*parts):
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:len(annos)]
            mmcv.dump(ordered_results, args.out)


if __name__ == '__main__':
    main()