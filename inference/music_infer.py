from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from audiocraft.data.audio import audio_write
import scipy
import torch


def music_infer(description, duration):
    model_id = 'AI-ModelScope/musicgen-large'
    sambert_hifigan_tts = pipeline(task=Tasks.text_to_speech, model=model_id, model_revision='v1.0.3')
    output = sambert_hifigan_tts(input=description.replace(',',' '), duration=duration)
    output_dict = output[OutputKeys.OUTPUT_WAV]
    wav = output_dict["wav"]
    sample_rate = output_dict["sample_rate"]
    audio_write(f'music', wav[0].cpu(), sample_rate, strategy="loudness")
    torch.cuda.empty_cache()
    return 'music.wav'