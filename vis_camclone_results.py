import pandas as pd
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import os

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

def combine_videos(csv_path, output_dir):
    """
    读取CSV文件，处理并拼接视频。

    Args:
        csv_path (str): CSV文件的路径。
        output_dir (str): 输出视频的保存目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    df = pd.read_csv(csv_path)
    # 初始化视频帧处理器，目标分辨率为 832x480
    processor = VideoFrameProcessor(height=480, width=832)

    for index, row in df.iterrows():
        ref_video_path = row['ref_video_path']
        gen_video_path = f"demo/camclone_i2v_output/camclone_output_video_{index:03d}.mp4"
        output_path = os.path.join(output_dir, f"output_{index}.mp4")

        # 检查视频文件是否存在
        if not os.path.exists(ref_video_path):
            print(f"警告: 左侧视频未找到，跳过: {ref_video_path}")
            continue
        if not os.path.exists(gen_video_path):
            print(f"警告: 右侧视频未找到，跳过: {gen_video_path}")
            continue

        print(f"正在处理第 {index} 个视频对...")

        cap_left = cv2.VideoCapture(ref_video_path)
        cap_right = cv2.VideoCapture(gen_video_path)

        # --- 读取左侧视频的所有帧并处理 ---
        left_frames = []
        while True:
            ret, frame = cap_left.read()
            if not ret:
                break
            left_frames.append(frame)

        # 帧数对齐：重复最后一帧，使总数达到81帧
        if left_frames:
            while len(left_frames) < 81:
                left_frames.append(left_frames[-1])
        
        # --- 设置视频写入器 ---
        fps = int(cap_right.get(cv2.CAP_PROP_FPS))
        right_width = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
        right_height = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 输出视频的宽度是处理后的左侧视频宽度(832) + 右侧视频宽度
        # 高度是两者中的最大值
        output_width = processor.width + right_width
        output_height = max(processor.height, right_height)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

        # --- 逐帧处理、拼接和写入 ---
        for i in range(81):
            ret_right, frame_right = cap_right.read()

            # 如果右侧视频提前结束，则停止处理
            if not ret_right:
                break
            
            # 从内存中获取左侧视频帧并处理
            frame_left_processed = processor.process_frame(left_frames[i])
            
            # 如果左右视频高度不同，需要将较矮的一个视频补上黑边
            if frame_left_processed.shape[0] != frame_right.shape[0]:
                # 在这里，我们假设处理后的左视频高度(480)和右视频高度一致或需要调整
                # 为简化起见，我们创建一个与输出高度一致的黑色背景
                combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                combined_frame[0:processor.height, 0:processor.width] = frame_left_processed
                combined_frame[0:right_height, processor.width:] = frame_right
            else:
                 # 水平拼接
                combined_frame = cv2.hconcat([frame_left_processed, frame_right])

            out_writer.write(combined_frame)
        
        print(f"已成功保存拼接视频: {output_path}")

        # 释放资源
        cap_left.release()
        cap_right.release()
        out_writer.release()

if __name__ == '__main__':
    # --- 配置区 ---
    # 请确保以下路径在您的环境中是正确的
    csv_file_path = "demo/example_csv/infer/example_camclone_testset.csv"
    output_directory = "demo/camera_ref&camclone_i2v_output"  # 您可以修改为您希望的输出目录
    
    # --- 运行主函数 ---
    combine_videos(csv_file_path, output_directory)