import sys

sys.path.append(".")
import numpy as np
from data.ph14t import Ph14TGeneralDataset, Ph14TMultiLinglDataset
from tqdm import tqdm
import cv2


if __name__ == "__main__":
    data_root = "dataset/PHOENIX-2014-T-release-v3"
    en_data_root = "large_files/ph14t_english"
    zh_data_root = "large_files/ph14t_chinese"
    mode = "train"

    # dataset = Ph14TGeneralDataset(data_root, mode)
    dataset = Ph14TMultiLinglDataset(data_root, zh_data_root, en_data_root, mode)

    i = 18  # 选择第一个样本
    data = dataset[i]
    print(data["video"].shape)
    frames = data["video"]

    # 设置视频参数
    fps = 30
    height, width = frames.shape[1], frames.shape[2]
    output_path = "outputs/output.mp4"

    # 创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码格式
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 将每一帧写入视频
    for frame in tqdm(frames):
        # 将帧转换为 uint8 类型
        frame = (frame * 255).astype(np.uint8)
        # 转换通道顺序从 RGB 到 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()  # 释放 VideoWriter 对象
