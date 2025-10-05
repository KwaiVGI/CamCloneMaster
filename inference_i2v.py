import sys
import torch
import torch.nn as nn
from diffsynth import WanVideoI2VPipeline, ModelManager, load_state_dict, save_video, VideoData
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import pandas as pd
import torchvision
from PIL import Image
import numpy as np
import json

from diffsynth.models.wan_video_dit import MLP, RMSNorm


class Image2VideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, infer=False):
        metadata = pd.read_csv(base_path)
        self.text = metadata["caption"].to_list()
        if infer == False:  # only for training
            self.path = metadata["video_path"].to_list()
        if "first_frame_path" in metadata.columns:  # first frame for infer
            self.first_frame_path = metadata["first_frame_path"].to_list()
        else:
            self.first_frame_path = None
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.infer = infer
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        self.image_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
        ])
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process, image_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            frame_indexs = list(range(num_frames))
            frame_indexs = [min(frame_index, reader.count_frames()-1) for frame_index in frame_indexs]
        else:
            frame_indexs = list(range(num_frames))
        
        frames = []
        first_frame = None
        for frame_id in frame_indexs:
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = image_process(frame)  # 输入必须是PIL
                first_frame = np.array(first_frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        # first_frame = frames[::, :1]  # C 1 H W
        return frames, first_frame


    def load_video(self, file_path):
        start_frame_id = 0
        frames, first_frame = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process, self.image_process)
        return frames, first_frame
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        frame = self.image_process(frame)
        frame = np.array(frame)
        return frame


    def __getitem__(self, data_id):
        while True:
            try:
                if self.infer:
                    text = self.text[data_id]
                    if self.first_frame_path is not None:
                        first_frame_path = self.first_frame_path[data_id]
                        first_frame = self.load_image(first_frame_path)
                    data = {"text": text, "first_frame": first_frame}
                    break
                else:
                    text = self.text[data_id]
                    path = self.path[data_id]
                    video, first_frame = self.load_video(path)
                    if self.first_frame_path is not None:
                        first_frame_path = self.first_frame_path[data_id]
                        first_frame = self.load_image(first_frame_path)
                    data = {"text": text, "video": video, "path": path, "first_frame": first_frame}
                    break
            except:
                data_id += 1
                data_id = data_id % (len(self.text))
        return data
    

    def __len__(self):
        return len(self.text)


def parse_args():
    parser = argparse.ArgumentParser(description="I2V Inference")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="demo/example_csv/infer/example_i2v_testset.csv",
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="models/CamCloneMaster-Wan2.1/Wan-I2V-1.3B-Step8000.ckpt",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="demo/i2v_output",
        help="Path to save the results.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    # 1. Load Wan2.1 pre-trained models
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    model_manager.load_models([
        "./models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        "./models/Wan-AI/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        "./models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        "./models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    ])
    pipe = WanVideoI2VPipeline.from_model_manager(model_manager)

    # 2. Initialize additional modules introduced in I2V
    with torch.no_grad():
        patch_embedding_ori = pipe.dit.patch_embedding
        patch_embedding_new = nn.Conv3d(36, 1536, kernel_size=(1, 2, 2), stride=(1, 2, 2))

        new_weights = torch.cat([
            patch_embedding_ori.weight,          # 第一轮复制 (16个通道)
            patch_embedding_ori.weight,          # 第二轮复制 (16个通道)
            patch_embedding_ori.weight[:, :4]    # 第三轮复制前4个通道
        ], dim=1) # 沿着输入通道维度(dim=1)拼接

        assert new_weights.shape[1] == 36
        # 赋值给新层
        patch_embedding_new.weight.copy_(new_weights)
        patch_embedding_new.bias.copy_(patch_embedding_ori.bias)

    pipe.dit.patch_embedding = patch_embedding_new

    dim=pipe.dit.blocks[0].self_attn.q.weight.shape[0]
    pipe.dit.img_emb = MLP(1280, dim)
    for block in pipe.dit.blocks:  # add for I2V
        block.cross_attn.k_img = nn.Linear(dim, dim)
        block.cross_attn.v_img = nn.Linear(dim, dim)
        block.cross_attn.norm_k_img = RMSNorm(dim, eps=1e-6)

    # 3. Load Adapted Wan-1.3B-I2V checkpoint
    state_dict = torch.load(args.ckpt_path, map_location="cpu")
    pipe.dit.load_state_dict(state_dict, strict=True)
    pipe.to("cuda")
    pipe.to(dtype=torch.bfloat16)

    output_dir = os.path.join(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    # 4. Prepare test data (source video, target camera, target trajectory)
    dataset = Image2VideoDataset(
        args.dataset_path,
        infer=True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )

    # 5. Inference
    for batch_idx, batch in enumerate(dataloader):
        target_text = batch["text"]
        input_image = batch["first_frame"]
        video = pipe(
            prompt=target_text,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            input_image=input_image,
            cfg_scale=args.cfg_scale,
            num_inference_steps=50,
            seed=0, tiled=True
        )
        save_video(video, os.path.join(output_dir, f"i2v_output_video_{batch_idx:03d}.mp4"), fps=30, quality=5)
    