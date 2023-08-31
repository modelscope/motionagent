from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
import torch
import gradio as gr


def i2v_infer_func(image_in):
    image_to_video_pipe = pipeline(task="image-to-video", model='damo/Image-to-Video', model_revision='v1.1.0')
    print(image_in)
    output_video_path = image_to_video_pipe(image_in, output_video='./i2v_output.mp4')[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    del image_to_video_pipe
    torch.cuda.empty_cache()
    return output_video_path


def i2v_infer(image_in):
    if image_in is None:
        raise gr.Error('请上传图片或等待图片上传完成(Please upload an image or wait for the image to finish uploading.)')
    return i2v_infer_func(image_in)


def v2v_infer_func(video_in, text_in):
    video_to_video_pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0')
    p_input = {
            'video_path': video_in,
            'text': text_in
        }
    output_video_path = video_to_video_pipe(p_input, output_video='./v2v_output.mp4')[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    del video_to_video_pipe
    torch.cuda.empty_cache()
    return output_video_path


def v2v_infer(video_in, text_in):
    if video_in is None:
        raise gr.Error('请先完成第一步(Please take the Step 1.)')
    if text_in is None:
        raise gr.Error('请输入文本描述(Please enter the vedio description.)')
    return v2v_infer_func(video_in, text_in)
