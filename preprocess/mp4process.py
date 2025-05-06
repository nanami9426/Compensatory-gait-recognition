import argparse
import torch
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join('..')))
from src.utils.window import Window
import logging
from ultralytics.utils import LOGGER
LOGGER.setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, required=True)
    # path为None则默认处理assets文件夹中所有内容
    parser.add_argument("--path", type=str, required=False)
    args = parser.parse_args()
    return args


args  =parse_args()
window_size = args.window_size
path = args.path if args.path is not None else './assets/'
detector = YOLO("../models/yolo11n-pose.pt")


def gen_df(video_path):
    window = Window(torch.device("cuda:0"), window_size, (17, 2))
    df = pd.DataFrame(np.random.rand(0, 17*2))
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    pbar = tqdm(total=total_frames, 
                desc=f"Processing Video {os.path.basename(video_path)}", 
                ncols=100, 
                bar_format="{desc}: |{bar}| {percentage:3.0f}% {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}]")
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        res = detector(frame)[0]
        kp = res.keypoints.xyn
        if len(res.boxes.cls) > 1:
            idx = res.boxes.conf.argmax(-1).item()
            kp = res.keypoints.xy[idx].unsqueeze(0)
        elif len(res.boxes.cls) == 0:
            kp = None
            pbar.update(1)
            continue
        ready = window.add(kp)
        if ready:
            data = pd.DataFrame(window.data.reshape(window_size, -1).cpu().numpy())
            window.clear()
            df = pd.concat([df, data], ignore_index=True)
        pbar.update(1)
    pbar.close()
    return df


def main(path):
    names = os.listdir(path)
    num_videos = 0
    for video_name in names:
        if not video_name.endswith('.mp4'):
            continue
        df = gen_df(os.path.join(path, video_name))
        csv_path = os.path.join('./rawdata', video_name[:-4] + '.csv')
        os.makedirs('rawdata', exist_ok=True)
        df.to_csv(csv_path)
        num_videos += 1
    print(f"Processed {num_videos} mp4 files, saved in {'./rawdata'}")


if __name__ == '__main__':
    main(path)