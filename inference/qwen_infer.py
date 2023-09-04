# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import time
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig

SCRIPT_TEMPLATE = """你是一个编剧，请根据提供的短片主题、背景、幕数、剧情要求，设计一个剧本。
主题：{theme}
背景：{background}
幕数：{act}幕
剧情要求：{scenario}
语言：{language}
剧本内容需要详细充分，每一幕不少于200字。"""

STORY_TEMPLATE = """我会给你一个简单的图片描述，请提供一个充满想象的虚构故事，故事内容需要与图像匹配。
主题：{story_theme}
图片描述：{picture}"""

STILL_TEMPLATE = """根据剧本内容，设计一张剧照的场景描述，能体现剧本的核心内容。
剧本：{script}
语言：{language}"""

SD_PROMPT_TEMPLATE = """我们现在要通过stable diffusion进行图片生成，请根据场景描述，提炼出用于文本生成图像的英文prompt。
示例：
描述：一只美丽的蝴蝶在花丛中翩翩起舞，翅膀上闪烁着五彩斑斓的光芒，引来了勤劳的蜜蜂。蜜蜂在蝴蝶身边绕来绕去，试图吸引蝴蝶的注意。蝴蝶终于注意到了蜜蜂，停下来停歇在花朵上，与蜜蜂对视。
prompt：butterfly dancing in flower field, wings shimmering with rainbow colors,  some bees flying around the butterfly, detailed realism, soft lighting, depth of field, 4k
描述：{still_description}
prompt："""

PROMPT_TEMPLATE = {
    "script": SCRIPT_TEMPLATE,
    "story":STORY_TEMPLATE,
    "still": STILL_TEMPLATE,
    "SD": SD_PROMPT_TEMPLATE,
}

def qwen_infer(inputs, history=None):
    tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", revision = 'v1.0.5',trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", revision = 'v1.0.5',device_map="auto", trust_remote_code=True,fp16 = True).eval()
    model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat",revision = 'v1.0.5', trust_remote_code=True)
    response, history = model.chat(tokenizer, inputs, history=history)
    torch.cuda.empty_cache()
    return response