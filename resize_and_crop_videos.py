import pandas as pd
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import os
import shutil

class VideoFrameProcessor:
    def __init__(self, height, width):
        """
        初始化处理器，设定目标高度和宽度。
        """
        self.height = height
        self.width = width

    def process_frame(self, frame_bgr):
        """
        对单帧图像执行缩放和中心裁剪。
        完全遵循用户提供的逻辑。

        Args:
            frame_bgr (np.array): 从OpenCV读取的BGR格式的视频帧。

        Returns:
            np.array: 处理完成后的BGR格式的视频帧。
        """
        # 1. 将OpenCV的BGR图像转换为PIL的RGB图像
        image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # 2. 实现 crop_and_resize 函数的逻辑
        img_width, img_height = image.size
        scale = max(self.width / img_width, self.height / img_height)
        
        # 使用torchvision.transforms.functional进行缩放
        resized_image = TF.resize(
            image,
            (round(img_height * scale), round(img_width * scale)),
            interpolation=TF.InterpolationMode.BILINEAR
        )

        # 3. 实现 v2.CenterCrop 的逻辑
        cropped_image = TF.center_crop(resized_image, (self.height, self.width))

        # 4. 将处理好的PIL图像转换回OpenCV的BGR格式
        final_frame_rgb = np.array(cropped_image)
        final_frame_bgr = cv2.cvtColor(final_frame_rgb, cv2.COLOR_RGB2BGR)

        return final_frame_bgr

def process_videos_from_csv(csv_path, video_path_column, height, width):
    """
    读取CSV文件，并对指定的视频列中的每个视频进行处理。

    Args:
        csv_path (str): CSV文件的路径。
        video_path_column (str): CSV文件中包含视频路径的列名。
        height (int): 目标视频高度。
        width (int): 目标视频宽度。
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误：找不到CSV文件 '{csv_path}'")
        return

    # 读取CSV文件
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    # 检查视频路径列是否存在
    if video_path_column not in df.columns:
        print(f"错误：CSV文件中找不到列 '{video_path_column}'")
        return
        
    # 初始化视频处理器
    processor = VideoFrameProcessor(height, width)

    # 遍历CSV中的每一行
    for index, row in df.iterrows():
        video_path = row[video_path_column]
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"警告：跳过不存在的视频文件 '{video_path}'")
            continue

        print(f"正在处理视频: {video_path}...")

        try:
            # 1. 读取原始视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"错误：无法打开视频文件 '{video_path}'")
                continue

            # 获取原始视频的编码格式
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # 2. 创建一个临时视频写入器，并将FPS设置为30
            temp_video_path = video_path + ".tmp.mp4"
            # ############################################################### #
            # ## 根据您的要求，这里的帧率（fps）被固定为 30。                 ## #
            # ############################################################### #
            out = cv2.VideoWriter(temp_video_path, fourcc, 30, (width, height))

            # 3. 逐帧处理
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 处理帧
                processed_frame = processor.process_frame(frame)
                
                # 写入新视频
                out.write(processed_frame)

            # 4. 释放资源
            cap.release()
            out.release()

            # 5. 原地替换：用处理后的视频覆盖原始视频
            shutil.move(temp_video_path, video_path)
            
            print(f"处理完成: {video_path}")

        except Exception as e:
            print(f"处理视频 '{video_path}' 时发生错误: {e}")
            # 如果出错，确保删除临时文件
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.remove(temp_video_path)

if __name__ == '__main__':
    # 1. CSV文件路径
    csv_path = 'demo/example_csv/infer/example_camclone_testset.csv'

    # 2. CSV文件中包含视频路径的列名
    video_path_column = 'ref_video_path'

    # 3. 目标视频的尺寸
    target_height = 480
    target_width = 832

    # 执行处理函数
    process_videos_from_csv(csv_path, video_path_column, target_height, target_width)

    print("\n所有视频处理完毕。")