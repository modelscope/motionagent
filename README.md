<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
    <h1>MotionAgent</h1>
<p>



# Introduction

如果您熟悉中文，可以阅读[中文版本的README](./README_ZH.md)。

MotionAgent is a deep learning model tool that can generate videos from user-created scripts. Users can create scripts, generate movie stills, generate images/videos, and compose background music through our provided toolset.

The model of MotionAgent is powered by the open-source model community [ModelScope](https://github.com/modelscope/modelscope).


# Features
- Script Generation
  - Users can generate scripts by specifying the story theme and background
  - The script generation model is based on LLM (such as Qwen-7B-Chat), which can generate scripts of various styles
- Movie still Generation
  - Generate corresponding movie still scene images 
- Video Generation
  - Generate videos from images
  - Support high-resolution video generation
- Music Generation
  - Custom style background music



# Quick Start

## Compatibility Verification
Verified environments:
- python3.8
- torch2.0.1
- CUDA11.7
- OS: Ubuntu 20.04
- Nvidia-A100 40G


## Resource Requirements
- GPU memory: 36GB
- Disk: It is recommended to reserve more than 50GB of storage space


## Installation Guide

### conda virtual environment

Use the conda virtual environment, refer to [Anaconda](https://docs.anaconda.com/anaconda/install/) to manage your dependencies, after installation, execute the following commands:

```shell
conda create -n motion_agent python=3.8
conda activate motion_agent

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/motionagent.git --depth 1
cd motionagent

# Install dependencies
pip3 install -r requirements.txt
pip3 install audiocraft==1.0.0

# Note: you can specify the pypi source to speed up the installation:
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run the application
python3 app.py

# Note: MotionAgent currently supports single-card GPU, if your environment has multiple cards, please use the following command
# CUDA_VISIBLE_DEVICES=0 python3 app.py
# Note: If you are using the Modelscope community Notebook or if your disk memory is less than 100GB, please turn on the clear_cache switch. Each run will result in re-downloading the model, causing a significant decrease in speed. Please be patient and wait.
# python3 app.py --clear_cache

# Finally, click on the URL generated in the log to access the page.
```


## Model List

[1]  Qwen-7B-Chat： [Model](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary)  |  [Space](https://modelscope.cn/studios/qwen/Qwen-7B-Chat-Demo/summary)

[2]  SDXL 1.0：[Model](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0/summary)  |  [Space](https://modelscope.cn/studios/AI-ModelScope/Stable_Diffusion_XL_1.0/summary)

[3]  I2VGen-XL： [Model](https://modelscope.cn/models/damo/Image-to-Video/summary)  |  [Space](https://modelscope.cn/models/damo/Video-to-Video/summary)

[4]  MusicGen： [Model](https://modelscope.cn/models/AI-ModelScope/musicgen-large/summary)  |  [Space](https://modelscope.cn/studios/AI-ModelScope/MusicGen/summary)


# More Information

- [ModelScope library](https://github.com/modelscope/modelscope/)

  ModelScope Library is a model ecosystem repository hosted on github, belonging to the Damo Academy Moda project.

- [Contribute models to ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
