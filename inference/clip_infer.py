from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = 'damo/cv_clip-interrogator'
pipe = pipeline(Tasks.image_captioning, model=model_id, model_revision='v1.0.0')

def clip_infer(input_image):
    picture = pipe(input_image)['caption']
    return picture