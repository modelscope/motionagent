# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch
import shutil
from modelscope.hub.utils.utils import get_cache_dir


def clip_infer(input_image):
    cache_dir = get_cache_dir()
    shutil.rmtree(cache_dir)
    model_id = 'damo/cv_clip-interrogator'
    pipe = pipeline(Tasks.image_captioning, model=model_id, model_revision='v1.0.0')
    picture = pipe(input_image)['caption']
    del pipe
    torch.cuda.empty_cache()
    return picture