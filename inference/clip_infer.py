from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torch


def clip_infer(input_image):
    model_id = 'damo/cv_clip-interrogator'
    pipe = pipeline(Tasks.image_captioning, model=model_id, model_revision='v1.0.0')
    picture = pipe(input_image)['caption']
    del pipe
    torch.cuda.empty_cache()
    return picture