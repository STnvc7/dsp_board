from typing import Callable, Literal
import math
import pytest
import torch
from dsp_board import Processor, set_enable_parallel, set_max_workers

@pytest.mark.torch
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("function", ["spectrogram", "mel_spectrogram", "mfcc"])
def test_torch_feature(
    audio_tensor: Callable[[int, str], torch.Tensor], 
    dsp_processor: Processor, 
    dim: int,
    device: Literal["cpu", "cuda"],
    function: str,
):
    if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")
            
    x = audio_tensor(dim, device)
    dsp_fn = getattr(dsp_processor, function)
    out = dsp_fn(x)
    
    n_bins = {
        "spectrogram": dsp_processor.fft_size // 2 + 1, 
        "mel_spectrogram": dsp_processor.n_mels, 
        "mfcc": dsp_processor.n_mfcc
    }
    
    for i in range(out.ndim - 2):
        assert out.shape[i] == x.shape[i]
    assert out.shape[-2] == n_bins[function]
    assert out.shape[-1] == math.ceil(x.shape[-1] / dsp_processor.hop_size)

@pytest.mark.sptk
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("function", ["mel_cepstrum", "mel_generalized_cepstrum"])
@pytest.mark.parametrize("enable_parallel, n_workers", [(False, 1), (True, 8)])
def test_sptk_feature(
    audio_tensor: Callable[[int, str], torch.Tensor], 
    dsp_processor: Processor, 
    dim: int,
    device: Literal["cpu", "cuda"],
    function: str,
    enable_parallel: bool,
    n_workers: int,
):
    set_enable_parallel(enable_parallel)
    set_max_workers(n_workers)
    
    if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")
            
    x = audio_tensor(dim, device)
    dsp_fn = getattr(dsp_processor, function)
    out = dsp_fn(x)
    
    n_bins = {
        "mel_cepstrum": dsp_processor.n_mcep + 1, 
        "mel_generalized_cepstrum": dsp_processor.n_mgc + 1, 
    }
    
    assert out.device == x.device
    for i in range(out.ndim - 2):
        assert out.shape[i] == x.shape[i]
    assert out.shape[-2] == n_bins[function]
    assert out.shape[-1] == math.ceil(x.shape[-1] / dsp_processor.hop_size)
    

@pytest.mark.world
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("function", ["pitch", "spectral_envelope", "aperiodicity", "band_aperiodicity", "vuv"])
@pytest.mark.parametrize("enable_parallel, n_workers", [(False, 1), (True, 8)])
def test_world_feature(
    audio_tensor: Callable[[int, str], torch.Tensor], 
    dsp_processor: Processor, 
    dim: int,
    device: Literal["cpu", "cuda"],
    function: str,
    enable_parallel: bool,
    n_workers: int,
):
    set_enable_parallel(enable_parallel)
    set_max_workers(n_workers)
    
    if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")
            
    x = audio_tensor(dim, device)
    dsp_fn = getattr(dsp_processor, function)
    out = dsp_fn(x)
    
    n_bins = {
        "pitch": 1, 
        "spectral_envelope": dsp_processor.fft_size // 2 + 1,
        "aperiodicity": dsp_processor.fft_size // 2 + 1,
        "band_aperiodicity": 1 if dsp_processor.sample_rate <= 16000 else (3 if dsp_processor.sample_rate <= 32000 else 5),
        "vuv": 1
    }
    
    for i in range(out.ndim - 2):
        assert out.shape[i] == x.shape[i]
    assert out.shape[-2] == n_bins[function]
    assert out.shape[-1] == math.ceil(x.shape[-1] / dsp_processor.hop_size)