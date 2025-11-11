import pytest
from typing import Literal
import os
import torchaudio
import torch
from dsp_board import Processor

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
SAMPLE_ROOT = f"{SCRIPT_DIR}/../jsut_sample"

@pytest.fixture(scope="session")
def dsp_processor():
    return Processor(
        sample_rate=48000,
        fft_size=1024,
        hop_size=256,
        window_size=1024,
        n_mels=80,
        n_mfcc=13,
        n_mcep=39,
        n_mgc=24,
        eps=1e-8,
        pitch_extract_method="harvest",
        trim_direction="forward",
        trim_trigger_level=7.0,
        trim_trigger_time=0.25,
        target_lufs=-24.0,
        target_db=-6.0,
    )

@pytest.fixture(scope="session")
def audio_tensor():
    def _audio_tensor(dim: Literal[1, 2, 3], device: Literal["cuda", "cpu"]):
        if dim == 1:
            x = audio_1d()
        elif dim == 2:
            x =  audio_2d()
        elif dim == 3:
            x = audio_3d()
            
        if device == "cuda":
            x = x.cuda()
        return x
    return _audio_tensor

def audio_1d():
    wav, _ = torchaudio.load(f"{SAMPLE_ROOT}/BASIC5000_0001.wav")
    wav = wav.squeeze(0)
    return wav

def audio_2d():
    x = [torchaudio.load(f"{SAMPLE_ROOT}/BASIC5000_000{i+1}.wav")[0] for i in range(0, 8)]
    offset = min([_x.shape[-1] for _x in x])
    x = torch.stack([_x[..., :offset].squeeze(0) for _x in x])
    return x

def audio_3d():
    x = [torchaudio.load(f"{SAMPLE_ROOT}/BASIC5000_000{i+1}.wav")[0] for i in range(0, 8)]
    offset = min([_x.shape[-1] for _x in x])
    x = torch.stack([_x[..., :offset].squeeze(0) for _x in x])
    x = x.reshape(4, 2, -1)
    return x