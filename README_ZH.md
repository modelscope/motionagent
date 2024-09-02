<p align="center">
    <br>
    <img src="https://modelscope.oss-cn-beijing.aliyuncs.com/modelscope.gif" width="400"/>
    <br>
    <h1>MotionAgent</h1>
<p>



# 介绍

MotionAgent是一个能将用户创造的剧本生成视频的深度学习模型工具。用户通过我们提供的工具组合，进行剧本创作、剧照生成、图片/视频生成、背景音乐谱写等工作。

MotionAgent的模型由[ModelScope](https://github.com/modelscope/modelscope)开源模型社区提供支持。


# 功能特性
- 剧本生成（Script Generation）
  - 用户指定故事主题和背景，即可生成剧本
  - 剧本生成模型基于LLM（如Qwen-7B-Chat），可生成多种风格的剧本
- 剧照生成（Movie still Generation）
  - 通过输入一幕剧本，即可生成对应的剧照场景图片
- 视频生成（Video Generation）
  - 图生视频
  - 支持高分辨率视频生成
- 音乐生成（Music Generation）
  - 自定义风格的背景音乐



# 快速开始

## 兼容性验证
已经验证过的环境：
- python3.8
- torch2.0.1
- CUDA11.7
- OS: Ubuntu 20.04
- Nvidia-A100 40G


## 资源要求
- GPU显存：36GB
- 磁盘: 推荐预留50GB以上的存储空间


## 安装指南

### conda虚拟环境

使用conda虚拟环境，参考[Anaconda](https://docs.anaconda.com/anaconda/install/)来管理您的依赖，安装完成后，执行如下命令：

```shell
conda create -n motion_agent python=3.8
conda activate motion_agent

GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/modelscope/motionagent.git --depth 1
cd motionagent

# 安装依赖
pip3 install -r requirements.txt
pip3 install audiocraft==1.0.0

# 备注：你可以指定pypi源来加速安装过程，例如：
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 运行应用
python3 app.py

# Note: MotionAgent目前支持单卡GPU，如果您的环境有多卡，请使用如下命令
# CUDA_VISIBLE_DEVICES=0 python3 app.py
# Note: 如果您使用了Modelscope社区的Notebook或者您的磁盘剩余内存小于100GB，推荐使用如下命令打开清理内存开关，每次运行都会重新下载模型导致速度变慢很多，请耐心等待
# python3 app.py --clear_cache

# 最后点击log中生成的URL即可访问页面。
```

              
## 模型列表

[1]  Qwen-7B-Chat： [模型](https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary)  |  [创空间](https://modelscope.cn/studios/qwen/Qwen-7B-Chat-Demo/summary)

[2]  SDXL 1.0：[模型](https://modelscope.cn/models/AI-ModelScope/stable-diffusion-xl-base-1.0/summary)  |  [创空间](https://modelscope.cn/studios/AI-ModelScope/Stable_Diffusion_XL_1.0/summary)

[3]  I2VGen-XL： [模型](https://modelscope.cn/models/damo/Image-to-Video/summary)  |  [创空间](https://modelscope.cn/models/damo/Video-to-Video/summary)

[4]  MusicGen： [模型](https://modelscope.cn/models/AI-ModelScope/musicgen-large/summary)  |  [创空间](https://modelscope.cn/studios/AI-ModelScope/MusicGen/summary)

                            

# 更多信息

- [ModelScope library](https://github.com/modelscope/modelscope/)

  ModelScope Library是一个托管于github上的模型生态仓库，隶属于达摩院魔搭项目。

- [贡献模型到ModelScope](https://modelscope.cn/docs/ModelScope%E6%A8%A1%E5%9E%8B%E6%8E%A5%E5%85%A5%E6%B5%81%E7%A8%8B%E6%A6%82%E8%A7%88)

# License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).

