# Copyright (c) Alibaba, Inc. and its affiliates.
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import shutil
from modelscope.hub.utils.utils import get_cache_dir


def music_infer(model_id, description, duration):
    cache_dir = get_cache_dir()
    shutil.rmtree(cache_dir)
    model = MusicGen.get_pretrained(model_id)
    model.set_generation_params(duration=duration)
    descriptions = [description]
    wav = model.generate(descriptions)
    for idx, one_wav in enumerate(wav):
        audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
    return '0.wav'
