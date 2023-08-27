from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

image_to_video_pipe = pipeline(task="image-to-video", model='damo/Image-to-Video', model_revision='v1.1.0')
video_to_video_pipe = pipeline(task="video-to-video", model='damo/Video-to-Video', model_revision='v1.1.0')

def i2v_infer(image_in):
    if image_in is None:
        raise gr.Error('请上传图片或等待图片上传完成(Please upload an image or wait for the image to finish uploading.)')
    print(image_in)
    output_video_path = image_to_video_pipe(image_in, output_video='./i2v_output.mp4')[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    return output_video_path


def v2v_infer(video_in, text_in):
    if video_in is None:
        raise gr.Error('请先完成第一步(Please take the Step 1.)')
    if text_in is None:
        raise gr.Error('请输入文本描述(Please enter the vedio description.)')
    p_input = {
            'video_path': video_in,
            'text': text_in
        }
    output_video_path = video_to_video_pipe(p_input, output_video='./v2v_output.mp4')[OutputKeys.OUTPUT_VIDEO]
    print(output_video_path)
    return output_video_path