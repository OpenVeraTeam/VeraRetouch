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

<h4>
<p align="center">
<a href="https://arxiv.org/abs/2604.27375">📄 arXiv Paper</a> &nbsp; 
<!-- <a href="waiting to write">🔗 Project Page</a> &nbsp; -->
</p>
</h4>
</div> 

## 🗓️ To Do List
- [x] Release inference code.
- [x] Release model weights.
- [ ] 🔴 Release iOS toy deployment.

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

### 🖥️ Inference
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

## 📲 Toy IOS depolyment
Comming soon...
