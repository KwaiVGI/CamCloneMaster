import copy
import os
import re
import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoI2VPipeline, ModelManager, load_state_dict
from diffsynth.models.wan_video_dit import MLP, RMSNorm
import torchvision
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.loggers import TensorBoardLogger

class CamCloneDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832):
        metadata = pd.read_csv(base_path)
        self.text = metadata["caption"].to_list()
        self.path = metadata["video_path"].to_list()
        self.ref_path = metadata["ref_video_path"].to_list()
        self.content_path = metadata["content_video_path"].to_list()
        if "first_frame_path" in metadata.columns:  # first frame for infer
            self.first_frame_path = metadata["first_frame_path"].to_list()
        else:
            self.first_frame_path = None
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
            
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
        # first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame


    def __getitem__(self, data_id):
        while True:
            try:
                text = self.text[data_id]
                path = self.path[data_id]
                ref_path = self.ref_path[data_id]
                content_path = self.content_path[data_id]
                video, first_frame = self.load_video(path)
                ref_video, _ = self.load_video(ref_path)
                content_video, _ = self.load_video(content_path)
                if self.first_frame_path is not None:
                    first_frame_path = self.first_frame_path[data_id]
                    first_frame = self.load_image(first_frame_path)
                data = {"text": text, "video": video, "path": path, "first_frame": first_frame, 'ref_video':ref_video, 'ref_path': ref_path, 'content_video': content_video, 'content_path': content_path}
                break
            except:
                data_id += 1
                data_id = data_id % (len(self.path))
        return data
    

    def __len__(self):
        return len(self.path)


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        text_encoder_path,
        vae_path,
        dit_path,
        image_encoder_path=None,
        tiled=False,
        tile_size=(34, 34),
        tile_stride=(18, 16),
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        resume_ckpt_path=None
    ):

        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_path = [text_encoder_path, vae_path, dit_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager.load_models(model_path)
        self.pipe = WanVideoI2VPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
        # self.pipe.dit.
        with torch.no_grad():
            patch_embedding_ori = self.pipe.dit.patch_embedding
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

        self.pipe.dit.patch_embedding = patch_embedding_new

        dim=self.pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        self.pipe.dit.img_emb = MLP(1280, dim)
        for block in self.pipe.dit.blocks:  # add for I2V
            block.cross_attn.k_img = nn.Linear(dim, dim)
            block.cross_attn.v_img = nn.Linear(dim, dim)
            block.cross_attn.norm_k_img = RMSNorm(dim, eps=1e-6)

        if resume_ckpt_path is not None:
            state_dict = torch.load(resume_ckpt_path, map_location="cpu")
            self.pipe.dit.load_state_dict(state_dict, strict=True)
            print(f"load ckpt from {resume_ckpt_path}!")

        self.freeze_parameters()
        for name, module in self.pipe.denoising_model().named_modules():
            if any(keyword in name for keyword in ["self_attn"]):
                print(f"Trainable: {name}")
                for param in module.parameters():
                    param.requires_grad = True

        trainable_params = 0
        seen_params = set()
        for name, module in self.pipe.denoising_model().named_modules():
            for param in module.parameters():
                if param.requires_grad and param not in seen_params:
                    trainable_params += param.numel()
                    seen_params.add(param)
        print(f"Total number of trainable parameters: {trainable_params}")
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        

    def training_step(self, batch, batch_idx):
        '''
        text_s: list [b]
        video_s: [b, c, f, h, w] [b, 3, 81, 480, 832]
        '''
        text_s, video_s, path_s, first_frame_s = batch['text'], batch['video'], batch['path'], batch['first_frame']
        ref_video_s, ref_path_s, content_video_s, content_path_s = batch['ref_video'], batch['ref_path'], batch['content_video'], batch['content_path']
        prompt_emb = self.pipe.encode_prompt(text_s)  # [b, n, c] [b, 512, 4096]
        self.pipe.device  = self.device
        # video
        video_s = video_s.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        latents = self.pipe.encode_video(video_s, **self.tiler_kwargs)  # [b, c, f, h, w]  [b, 16, 21, 60, 104])]
        ref_latents = self.pipe.encode_video(ref_video_s, **self.tiler_kwargs)
        content_latents = self.pipe.encode_video(content_video_s, **self.tiler_kwargs)
        ref_latents = F.pad(ref_latents, (0, 0, 0, 0, 0, 0, 0, 20))
        content_latents = F.pad(content_latents, (0, 0, 0, 0, 0, 0, 0, 20))
        _, _, num_frames, height, width = video_s.shape
        first_frame_s = [Image.fromarray(first_frame.cpu().numpy()) for first_frame in first_frame_s]
        image_emb = self.pipe.encode_image(first_frame_s, num_frames, height, width)

        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        origin_latents = copy.deepcopy(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)

        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)
        
        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, cam_emb=None, **prompt_emb, **extra_input, **image_emb,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
            ref_latents = ref_latents,
            content_latents = content_latents,
        )
        loss = torch.nn.functional.mse_loss(noise_pred[:, :, :21, ...].float(), training_target[:, :, :21, ...].float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = self.trainer.checkpoint_callback.dirpath
        print(f"Checkpoint directory: {checkpoint_dir}")
        current_step = self.global_step
        print(f"Current step: {current_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, f"step{current_step}.ckpt"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train CamCloneMaster")
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="CameraClone-Dataset/CamCloneDataset.csv/",
        help="The path of the Dataset.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="models/train",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default="models/Wan-AI/Wan2.1-T2V-1.3B/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        help="Path of image encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="deepspeed_stage_1",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=True,
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--metadata_file_name",
        type=str,
        default="metadata.csv",
    )
    parser.add_argument(
        "--resume_ckpt_path",
        type=str,
        default="models/CamCloneMaster-Wan2.1/Wan-I2V-1.3B-Step8000.ckpt",
    )
    args = parser.parse_args()
    return args
    
    
def train(args):
    dataset = CamCloneDataset(
        args.dataset_path,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=4,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        learning_rate=args.learning_rate,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        resume_ckpt_path=args.resume_ckpt_path,
    )
    

    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1, every_n_train_steps=500,filename='{epoch}-{step}')],
        logger=logger
        )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    train(args)
