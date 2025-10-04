<p align="center" >
    <img src="Figs/CamCloneMaster_Logo.png"  width="70%" >
</p>

# <div align="center" >Enabling Reference-based Camera Control for Video Generation<div align="center">

###  <div align="center"> SIGGRAPH Asia 2025 </div>
<div align="center">
  <a href="https://luo0207.github.io/yawenluo/">Yawen Luo</a>, 
  <a href="https://jianhongbai.github.io/">Jianhong Bai</a>, 
  <a href="https://xiaoyushi97.github.io/">Xiaoyu Shi</a><sup>†</sup>, 
  <a href="https://menghanxia.github.io/">Menghan Xia</a>, 
  <a href="https://xinntao.github.io/">Xintao Wang</a>, 
  <a href="https://magicwpf.github.io/">Pengfei Wan</a>, 
  <a href="https://openreview.net/profile?id=~Di_ZHANG3">Di Zhang</a>,
  <a href="https://openreview.net/profile?id=~Kun_Gai1">Kun Gai</a>,
  <a href="https://tianfan.info/">Tianfan Xue</a><sup>†</sup>
</div>

<br>

<p align="center">
  <a href='https://camclonemaster.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  &nbsp;
  <a href='https://youtu.be/Os18zynOqM4'><img src='https://img.shields.io/static/v1?label=Youtube&message=DemoVideo&color=yellow&logo=youtube'></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2506.03140"><img src="https://img.shields.io/static/v1?label=Arxiv&message=CamCloneMaster&color=red&logo=arxiv"></a>
  &nbsp;
  <a href='https://huggingface.co/datasets/KwaiVGI/CameraClone-Dataset'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange'></a>
</p>

**Note:** This open-source repository is intended to provide a reference implementation. Due to the difference in the underlying I2V model's performance, the open-source version may not achieve the same performance as the model in our paper. 

## 🔥 Updates
- __[2025.09.25]__: [CamCloneMaster](https://arxiv.org/abs/2506.03140) has been accepted by SIGGRAPH Aisa 2025.
- __[2025.09.08]__: [CameraClone Dataset](https://huggingface.co/datasets/KwaiVGI/CameraClone-Dataset/) is avaliable.
- __[2025.06.03]__: Release the [Project Page](https://camclonemaster.github.io/) and the [Arxiv](https://arxiv.org/abs/2506.03140) version.

## 📷 Introduction
**TL;DR:** We propose CamCloneMaster, a framework that enables users to replicate camera movements from reference videos without requiring camera parameters or test-time fine-tuning. CamCloneMaster seamlessly supports reference-based camera control for both I2V and V2V tasks within a unified framework. We also release our [CameraClone Dataset](https://huggingface.co/datasets/KwaiVGI/CameraClone-Dataset) rendered with Unreal Engine 5.

<div align="center">

[![Watch the video](Figs/DemoFirstPageWithButton.png)](https://www.youtube.com/watch?v=Os18zynOqM4)

</div>

## ⚙️ Code: CamCloneMaster + Wan2.1 (Inference & Training)

The model utilized in our paper is an internally developed T2V model, not [Wan2.1](https://github.com/Wan-Video/Wan2.1). Due to company policy restrictions, we are unable to open-source the model used in the paper. 

Due to training cost limitations, we adapted the [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) model for Image-to-Video (I2V) generation. This was achieved by conditioning the first frame through channel concatenation, a method proposed in the [Wan technical report]((https://arxiv.org/abs/2503.20314)), rather than using the larger [Wan2.1-I2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) model. We then integrated CamCloneMaster with this adapted 1.3B model to validate our method's effectiveness. Please note that results may differ from the demo due to this difference in the underlying I2V model.

### Inference

####  Step 1: Set up the environment

[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) requires Rust and Cargo to compile extensions. You can install them using the following command:

```bash
curl --proto '=https' --tlsv1.2 -sSf [https://sh.rustup.rs](https://sh.rustup.rs/) | sh
. "$HOME/.cargo/env"
```

Install [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio):
```bash
git clone https://github.com/KwaiVGI/CamCloneMaster
cd CamCloneMaster
pip install -e .
```

####  Step 2: Download the pretrained checkpoints
1. Download the pre-trained Wan2.1 models
```bash
cd CamCloneMaster
python download_wan2.1.py
```

2. Download the adapted Wan2.1-I2V-1.3B models
