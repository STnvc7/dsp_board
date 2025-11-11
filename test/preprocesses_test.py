from typing import Callable, Literal
import pytest

import torch
import torchaudio
from dsp_board import Processor

@pytest.mark.torch
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("target_sample_rate", [8000, 16000, 32000])
def test_resample(
    audio_tensor: Callable[[int, str], torch.Tensor], 
    dsp_processor: Processor, 
    dim: int,
    device: Literal["cpu", "cuda"],
    target_sample_rate: int
):
    if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")
            
    x = audio_tensor(dim, device)
    out = dsp_processor.resample(x, target_sample_rate)
    
    for i in range(out.ndim - 1):
        assert out.shape[i] == x.shape[i]
    assert out.shape[-1] < x.shape[-1]
    
    
@pytest.mark.torch
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("direction", ["forward", "backward", "both"])
def test_trim(
    audio_tensor: Callable[[int, str], torch.Tensor], 
    dsp_processor: Processor, 
    dim: int,
    device: Literal["cpu", "cuda"],
    direction: Literal["forward", "backward", "both"]
):
    if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")
            
    x = audio_tensor(dim, device)
    out = dsp_processor.trim(x, direction=direction)
    
    for i in range(out.ndim - 1):
        assert out.shape[i] == x.shape[i]
    assert out.shape[-1] < x.shape[-1]


@pytest.mark.torch
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("target_db", [-10.0, -6.0, -1.0])
def test_peak_normalize(
    audio_tensor: Callable[[int, str], torch.Tensor], 
    dsp_processor: Processor, 
    dim: int,
    device: Literal["cpu", "cuda"],
    target_db: float
):
    if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")
            
    x = audio_tensor(dim, device)
    out = dsp_processor.peak_normalize(x, target_db=target_db)
    
    assert out.shape == x.shape
    assert out.device == x.device
    
    out_flat = out[None,:] if dim == 1 else out.view(-1, out.shape[-1])
    peaks_out = 20 * torch.log10(out_flat.abs().max(dim=-1)[0])
    assert torch.allclose(peaks_out, torch.full_like(peaks_out, target_db))

@pytest.mark.torch
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("target_lufs", [-24, -20])
def test_loudness_normalize(
    audio_tensor: Callable[[int, str], torch.Tensor], 
    dsp_processor: Processor, 
    dim: int,
    device: Literal["cpu", "cuda"],
    target_lufs: float
):
    if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")
            
    x = audio_tensor(dim, device)
    out = dsp_processor.loudness_normalize(x, target_lufs=target_lufs)
    
    assert out.shape == x.shape
    
    out_flat = out[None,:] if dim == 1 else out.view(-1, out.shape[-1])
    loudness_out = []
    for i in range(out_flat.shape[0]):
        _loudness = torchaudio.functional.loudness(out_flat[i].unsqueeze(0), dsp_processor.sample_rate)
        loudness_out.append(_loudness.squeeze(0))
    loudness_out = torch.stack(loudness_out, dim=0)
    assert torch.allclose(loudness_out, torch.full_like(loudness_out, target_lufs))
