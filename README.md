<div align="center">


<h1>VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching</h1>

<div>
    <a href="https://Apollo-Yi.github.io" target="_blank">Yihong Guo</a><sup>1</sup>&emsp;
    <a href="https://youweilyu.github.io/" target="_blank">Youwei Lyu</a><sup>2</sup>&emsp;
    <a href="https://me.jeffreet.com/" target="_blank">Jiajun Tang</a><sup>2</sup>&emsp;
    <a href="https://github.com/OpenVeraTeam/VeraRetouch" target="_blank">Yizhuo Zhou</a><sup>1</sup>&emsp;
    <a href="https://github.com/OpenVeraTeam/VeraRetouch" target="_blank">Hongliang Wang</a><sup>3</sup>&emsp;
    <a href="https://scholar.google.com/citations?hl=zh-CN&user=Pcsml4oAAAAJ" target="_blank">Jinwei Chen</a><sup>2</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=kj5HiGgAAAAJ&hl=en" target="_blank">Changqing Zou</a><sup>1†</sup>&emsp;
    <a href="https://fqnchina.github.io/" target="_blank">Qingnan Fan</a><sup>2</sup>
</div>
<div>
    <sup>1</sup>Zhejiang University, <sup>2</sup>vivo BlueImage Lab, <sup>3</sup>University of Chinese Academy of Sciences
</div>

&nbsp;&nbsp;<br>
<a href='https://arxiv.org/abs/2604.27375'><img src='https://img.shields.io/badge/arXiv-2604.27375-b31b1b.svg'></a> &nbsp;&nbsp;
<a href='https://apollo-yi.github.io/VeraRetouch/'><img src='https://img.shields.io/badge/Project-Page-green'></a> &nbsp;
</div> 

## 🗓️ To Do List
- [x] Release VeraRetouch inference code.
- [x] Release VeraRetouch model weights.
- [x] Release Retouch Encoder-Renderer inference code and weights.
- [ ] 🔴 Release iOS toy deployment.

## 🌟 Highlights
* 🔥 Lightweight design for controllable, interpretable mobile deployment.
* 🔥 Free-resolution input for flexible retouching across diverse image sizes.
* 🔥 Fully differentiable renderer for direct pixel-level training.
* 🔥 Unified support for auto, style, and parameter retouching.
* 🔥 AetherRetouch-1M+ for large-scale professional supervision.

<!-- <table>
<tr>
    <td><img src="assets/Auto.gif" alt="Auto Mode"></td>
    <td><img src="assets/Style.gif" alt="Style Mode"></td>
    <td><img src="assets/Param.gif" alt="Param Mode"></td>
</tr>
</table> -->
<table>
<tr>
    <td align="center">
        <img src="assets/Auto.gif" alt="Auto Mode"><br>
        <strong>Auto Mode</strong>
    </td>
    <td align="center">
        <img src="assets/Style.gif" alt="Style Mode"><br>
        <strong>Style Mode</strong>
    </td>
    <td align="center">
        <img src="assets/Param.gif" alt="Param Mode"><br>
        <strong>Param Mode</strong>
    </td>
</tr>
</table>
<p align="center">(Demo videos play at 3x speed)</p>

## 🎬 <a name="overview"></a>Overview

![overview](assets/Model-Structure.png)

Reasoning photo retouching has gained significant traction, requiring models to analyze image defects, give reasoning processes, and execute precise retouching enhancements. However, existing approaches often rely on non-differentiable external software, creating optimization barriers and suffering from high parameter redundancy and limited generalization. To address these challenges, we propose VeraRetouch, a lightweight and fully differentiable framework for multi-task photo retouching. We employ a 0.5B Vision-Language Model (VLM) as the central intelligence to formulate retouching plans based on instructions and scene semantics. Furthermore, we develop a fully differentiable Retouch Renderer that replaces external tools, enabling direct end-to-end pixel-level training through decoupled control latents for lighting, global color, and specific color adjustments. To overcome data scarcity, we introduce AetherRetouch-1M+, the first million-scale dataset for professional retouching, constructed via a new inverse degradation workflow. Furthermore, we propose DAPO-AE, a reinforcement learning post-training strategy that enhances autonomous aesthetic cognition. Extensive experiments demonstrate that VeraRetouch achieves state-of-the-art performance across multiple benchmarks while maintaining a significantly smaller footprint, enabling mobile deployment.

## 🚀 Quick Start

### ⚙️ Environment
```bash
# Clone the repository
git clone https://github.com/OpenVeraTeam/VeraRetouch.git
cd VeraRetouch

# Create and activate conda environment
conda create -n vera-retouch python=3.10
conda activate vera-retouch
pip install -r requirements.txt
```

### ☕ Pretrained Model
Download our pretrained weights from [HuggingFace](https://huggingface.co/Gyh68/VeraRetouch/tree/main).

You can put the pretrained model to ./checkpoints

If you want to try "Reference Retouch" of Retouch Encoder-Renderer. please download Encoder-Renderer pretrained weights from [this HuggingFace link](https://huggingface.co/Gyh68/VeraRetouch.Encoder_Renderer/blob/main/encoder_renderer.pth).

### 🎨 VeraRetouch Inference
Our model supports three inference modes:

- Auto Retouch: Only an image is input.
```bash
python inference.py --mode auto \
                    --model-path ./checkpoints/VeraRetouch    # the pretrained model path \
                    --img_paths ./data_samples/input/sample_flower.jpg    # input image paths, multiple paths are supported \
                    --save_dir ./data_samples/output/    # output texts and images save path \
                    --chunk -1    # Enable when GPU memory is insufficient. The renderer will process large images in chunks. Recommended value: 262144 (512*512), enabling chunking will reduce inference speed. \
                    --batch_size 1    # Support batch inference
```
- Style Retouch: An image and user prompt are input.
```bash
python inference.py --mode style \
                    --prompt "I want a dreamy bright pink style."    # style user prompt(only 'style' mode used) \
                    --model-path ./checkpoints/VeraRetouch    # the pretrained model path \
                    --img_paths ./data_samples/input/sample_flower.jpg    # input image paths, multiple paths are supported \
                    --save_dir ./data_samples/output/    # output texts and images save path \
                    --chunk -1    # Enable when GPU memory is insufficient. The renderer will process large images in chunks. Recommended value: 262144 (512*512), enabling chunking will reduce inference speed. \
                    --batch_size 1    # Support batch inference
```
- Param Retouch: An image and retouching operator parameters are input.
```bash
python inference.py --mode style \
                    --instruction_path ./data_samples/param.json    # retourch operator parameters(only 'param' mode used) \
                    --model-path ./checkpoints/VeraRetouch    # the pretrained model path \
                    --img_paths ./data_samples/input/sample_flower.jpg    # input image paths, multiple paths are supported \
                    --save_dir ./data_samples/output/    # output texts and images save path \
                    --chunk -1    # Enable when GPU memory is insufficient. The renderer will process large images in chunks. Recommended value: 262144 (512*512), enabling chunking will reduce inference speed. \
                    --batch_size 1    # Support batch inference
```

### 🖥️ Retouch Encoder-Renderer Inference
The Retouch Encoder-Renderer enables image retouching with reference based on either a pair of retouching images or a single target retouching image.
- Reference-based retouching with a pair of retouching images
```bash
python infer_ref_retouch.py --pretrained_path ./checkpoints/encoder_renderer.pth    # Path to the pretrained model weights \
                            --output_dir ./data_samples/ref_outputs    # Directory to save the final retouched output images \
                            --ref_before_img_path ./data_samples/ref_inputs/ref/before.jpg   # File path of the original unretouched reference image \
                            --ref_after_img_path ./data_samples/ref_inputs/ref/after.jpg    # File path of the retouched reference target image \
                            --input_img_path ./data_samples/ref_inputs/sample.jpg    # File path of the input image to be retouched \
                            --chunk -1    # Enable when GPU memory is insufficient. The renderer will process large images in chunks. Recommended value: 262144 (512*512), enabling chunking will reduce inference speed. \
```
- Reference-based retouching with a single target retouching image (referencing the processing paradigm of paper [InstantRetouch: Personalized Image Retouching without Test-time Fine-tuning Using an Asymmetric Auto-Encoder](https://arxiv.org/abs/2602.17044): replace the pre-retouching image in the reference image pair with the input image)
```bash
python infer_ref_retouch.py --pretrained_path ./checkpoints/encoder_renderer.pth    # Path to the pretrained model weights \
                            --output_dir ./data_samples/ref_outputs    # Directory to save the final retouched output images \
                            --ref_before_img_path ./data_samples/ref_inputs/sample.jpg   # !!! Keep same with input_img_path.!!! \
                            --ref_after_img_path ./data_samples/ref_inputs/ref/after.jpg    # File path of the retouched reference target image \
                            --input_img_path ./data_samples/ref_inputs/sample.jpg    # File path of the input image to be retouched \
                            --chunk -1    # Enable when GPU memory is insufficient. The renderer will process large images in chunks. Recommended value: 262144 (512*512), enabling chunking will reduce inference speed. \
```

## 📲 iOS and macOS depolyment
We have released the macOS and iOS deployment demos! Please follow the step-by-step instructions below.

### 1. Download ML Model Weights

We have released the Core ML converted model weights. Please download the appropriate version from Hugging Face:

| Version | Description | Hugging Face Link |
|---|---|---|
| Without Quantization | Full-precision Core ML model with better performance | [Gyh68/ml-VeraRetouch](https://huggingface.co/Gyh68/ml-VeraRetouch/tree/main) |
| INT8 Quantized | INT8 quantized Core ML model with smaller size and faster inference | [Gyh68/ml-VeraRetouch-int8](https://huggingface.co/Gyh68/ml-VeraRetouch-int8/tree/main) |

> **Note:** The INT8 quantized model may lead to some performance degradation.

### 2. Move the downloaded weights directory

After downloading the model weights, move the directory into the `./ml-veraretouch/VeraRetouchCore` and rename it to `model`:

```bash
cd VeraRetouch
mv <downloaded_weights_dir> ./ml-veraretouch/VeraRetouchCore/model
```

### 3. Prepare the Xcode App and Open the Project

Please install Xcode on your Mac, then open the `./ml-veraretouch` project in Xcode.

Update the package dependencies as follows:

| Package | Repository | Dependency Rule | Version |
|---|---|---|---|
| `mlx-swift` | `https://github.com/ml-explore/mlx-swift` | Up to Next Major Version | `0.21.2` |
| `mlx-libraries` | `https://github.com/ml-explore/mlx-swift-examples` | Exact Version | `2.21.2` |
| `swift-transformers` | `https://github.com/huggingface/swift-transformers` | Exact Version | `0.1.18` |

### 4. Build and Run

Finally, select your target device in Xcode, then build and run the project.

> **Note:** The project has been successfully tested on MacBook Air (M4) and iPhone 13 Pro Max.

## 🎁 What’s New in iOS & macOS Deployment?

1. We introduce **3D LUT acceleration** to optimize the performance of the Retouch Renderer.  
   Specifically, we first use the Retouch Renderer to generate a 3D LUT, and then apply the LUT for image processing.  
   This significantly improves efficiency for high-resolution images with minimal quality loss.

2. **Live Photo support is now available!**  
   You can upload Live Photos for preview, select a reference frame, and the system will apply retouching consistently across the entire Live Photo sequence.
<p align="center">
  <img src="./assets/LivePhoto.gif" width="400"/>
</p>

> **Note:** If you need to reselect the key photo, we recommend trying [LiveMoments](https://github.com/OpenVeraTeam/LiveMoments).

3. We have integrated **Reference Retouch directly into the app**.  
   You can now perform Ref-Retouch operations directly within the dedicated interface.

## 📘 License
```
The code is licensed under Apache License 2.0.

The model weights are released for academic research purposes only.  
Commercial use of the model weights or any derived models is strictly prohibited.
```

## <a name="citation"></a>🎓 Citation
```
@article{guo2026veraretouch,
  title={VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching},
  author={Guo, Yihong and Lyu, Youwei and Tang, Jiajun and Zhou, Yizhuo and Wang, Hongliang and Chen, Jinwei and Zou, Changqing and Fan, Qingnan},
  journal={arXiv preprint arXiv:2604.27375},
  year={2026}
}
```
