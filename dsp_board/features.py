import math
from typing import Literal, Optional

import torch
import torchaudio
import numpy as np

from dsp_board.transforms import stft
from dsp_board import core
from dsp_board.utils.channelize import channelize
from dsp_board.utils.tensor import to_numpy, from_numpy, fix_length
from dsp_board.utils.parallelize import get_enable_parallel, set_enable_parallel

ALPHA = {
    8000: 0.312,
    12000: 0.369,
    16000: 0.410,
    22050: 0.455,
    24000: 0.466,
    32000: 0.504,
    44100: 0.544,
    48000: 0.554,
}

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def spectrogram(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    window_size: Optional[int] = None,
    power: bool = False,
    log: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the amplitude spectrogram from an input waveform using STFT.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        fft_size (int): Number of FFT points (i.e., window length for FFT).
        hop_size (int): Number of audio samples between adjacent STFT columns (frame shift).
        window_size (Optional[int], optional): Length of the window function. If None, defaults to fft_size.
        power (bool, optional): If True, returns power spectrogram (magnitude squared). If False, returns magnitude spectrogram. Defaults to True.
        log (bool, optional): If True, applies logarithmic scaling to the output spectrogram. Defaults to False.
        eps (float, optional): Small constant added to avoid log(0). Defaults to 1e-8.

    Returns:
        torch.Tensor: Amplitude spectrogram with shape (B?, fft_size//2+1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    amp = stft(x, fft_size, hop_size, window_size)
    amp = amp.abs().pow(2).clip(min=eps)
    if power is False:
        amp = amp.sqrt()
    if log:
        amp = amp.log()
        
    amp = fix_length(amp, math.ceil(x.shape[-1] / hop_size), -1)
    
    return amp

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def mel_spectrogram(
    x: torch.Tensor,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    n_mels: int = 80,
    window_size: Optional[int] = None,
    power: bool = False,
    log: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the mel-scaled spectrogram from an input waveform.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        n_mels (int): Number of mel filter banks to use.
        window_size (Optional[int], optional): Window length for STFT. If None, defaults to `fft_size`.
        log (bool, optional): Whether to apply log scaling to the mel spectrogram. Defaults to True.
        eps (float, optional): Small constant added before taking log to avoid log(0). Defaults to 1e-8.

    Returns:
        torch.Tensor: Mel spectrogram with shape (B?, n_mels, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    spc = spectrogram(x, fft_size, hop_size, window_size, power=power, log=False)
    mel_filter = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        n_stft=(fft_size // 2 + 1),
        norm="slaney",
        mel_scale="slaney",
    ).to(x.device)
    mel_spc = mel_filter(spc)

    if log:
        mel_spc = torch.clamp(input=mel_spc, min=eps)
        mel_spc = torch.log(mel_spc)
        
    mel_spc = fix_length(mel_spc, math.ceil(x.shape[-1] / hop_size), -1)
    
    return mel_spc
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def linear_energy(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    window_size: Optional[int] = None,
    power: bool = False,
    log: bool = True,
    eps: float = 1e-8,
    reduction: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    """
    Computes the per-frame energy from a linear spectrogram.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        fft_size (int): Number of FFT points (i.e., window length for FFT).
        hop_size (int): Number of audio samples between adjacent STFT columns (frame shift).
        window_size (Optional[int], optional): Length of the window function. If None, defaults to fft_size.
        power (bool, optional): If True, returns power spectrogram (magnitude squared). If False, returns magnitude spectrogram. Defaults to True.
        log (bool, optional): If True, applies logarithmic scaling to the output spectrogram. Defaults to False.
        eps (float, optional): Small constant added to avoid log(0). Defaults to 1e-8.
        reduction (Literal["sum", "mean"], optional): Reduction method to apply to the spectrogram. Defaults to "sum".

    Returns:
        torch.Tensor: Per-frame energy with shape (B?, 1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    spc = spectrogram(x, fft_size, hop_size, window_size, power, log, eps)
    
    if reduction == "sum":
        energy = torch.sum(spc, dim=-2, keepdim=True)
    elif reduction == "mean":
        energy = torch.mean(spc, dim=-2, keepdim=True)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    
    return energy
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def mel_energy(
    x: torch.Tensor,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    n_mels: int = 80,
    window_size: Optional[int] = None,
    power: bool = False,
    log: bool = True,
    eps: float = 1e-8,
    reduction: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    """
    Computes the per-frame energy from a mel spectrogram.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        n_mels (int): Number of mel filter banks to use.
        window_size (Optional[int], optional): Window length for STFT. If None, defaults to `fft_size`.
        log (bool, optional): Whether to apply log scaling to the mel spectrogram. Defaults to True.
        eps (float, optional): Small constant added before taking log to avoid log(0). Defaults to 1e-8.
        reduction (Literal["sum", "mean"], optional): Reduction method to apply to the spectrogram. Defaults to "sum".
        
    Returns:
        torch.Tensor: Per-frame energy with shape (B?, 1, T).
            - B? is the same as the input.
            - T is the number of time frames.
    """
    mel_spc = mel_spectrogram(x, sample_rate,fft_size, hop_size, n_mels, window_size, power, log, eps)
    
    if reduction == "sum":
        energy = torch.sum(mel_spc, dim=-2, keepdim=True)
    elif reduction == "mean":
        energy = torch.mean(mel_spc, dim=-2, keepdim=True)
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
    
    return energy

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def mel_cepstrum(
    x: torch.Tensor,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    order: int = 39,
    pitch_extract_method: Literal["dio", "harvest"] = "harvest",
):
    """
    Compute the mel-cepstrum from a waveform using WORLD's spectral envelope.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        order (int): Order of the mel-cepstrum. Defaults to 39.
        method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        torch.Tensor: Mel cepstrum with shape (B?, order+1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    device = x.device
    alpha = ALPHA.get(sample_rate, 0.466)
    
    x_np = to_numpy(x, np.float64)
    sp = core.world.spectral_envelope(
        x_np, 
        sample_rate=sample_rate, 
        fft_size=fft_size, 
        hop_size=hop_size, 
        pitch_extract_method=pitch_extract_method
    )
    mcep = core.sptk.mel_cepstrum(
        sp, 
        order=order, 
        alpha=alpha
    )
    mcep = from_numpy(mcep, device=device, torch_dtype=torch.float32)
    mcep = mcep.permute(0,2,1)
    mcep = fix_length(mcep, math.ceil(x.shape[-1] / hop_size), -1)
    
    return mcep

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def mel_generalized_cepstrum(
    x: torch.Tensor,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    order: int = 24,
    stage: int = 5,
):
    """
    Compute the Mel-Generalized Cepstrum (MGC) from an input waveform.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sample rate of the input waveform.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        order (int, optional): Order of the Mel-generalized cepstrum. Defaults to 24.
        stage (int, optional): Gamma parameter (gamma = -1/stage) for MGC calculation.
            Defaults to 5.

    Returns:
        torch.Tensor: Mel generalized cepstrum with shape (B?, order+1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    device = x.device
    alpha = ALPHA.get(sample_rate, 0.466)
    gamma = -1.0 / stage
    
    MAX_WAV_VALUE = 32768.0
    frames = x.unfold(dimension=1, size=fft_size, step=hop_size)
    blackman = torch.blackman_window(fft_size, periodic=True, dtype=x.dtype, device=x.device)
    frames = frames * blackman.unsqueeze(0)
    
    frames = to_numpy(frames*MAX_WAV_VALUE, np.float64)

    # WARNING: sptk.mgcep slows down with threads. Do not parallelize.
    is_enable = get_enable_parallel()
    set_enable_parallel(False)
    mgc = core.sptk.mel_generalized_cepstrum(
        frames, 
        order=order, 
        alpha=alpha, 
        gamma=gamma
    )
    set_enable_parallel(is_enable)
    
    mgc = from_numpy(mgc, device=device, torch_dtype=torch.float32)
    mgc = mgc.permute(0, 2, 1)
    mgc = fix_length(mgc, math.ceil(x.shape[-1] / hop_size), -1)
    
    return mgc


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def mfcc(
    x: torch.Tensor,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    n_mels: int,
    n_mfcc: int = 13,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the Mel-Frequency Cepstral Coefficients (MFCCs) from an audio signal.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        n_mels (int): Number of mel filter banks to use.
        n_mfcc (int, optional): Number of MFCC coefficients to compute. Defaults to 13.
        eps (float, optional): Small value added to the mel-spectrogram to avoid log of zero. Defaults to 1e-8.

    Returns:
        torch.Tensor: Mel frequency cepstral coefficients with shape (B?, n_mfcc, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    mel_spc = mel_spectrogram(
        x,
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_size=hop_size,
        n_mels=n_mels,
        log=True,
        eps=eps,
    )
    dct = torchaudio.functional.create_dct(
        n_mfcc=n_mfcc, n_mels=n_mels, norm="ortho"
    ).to(x.device)
    mfcc = torch.matmul(mel_spc.transpose(-1, -2), dct)
    
    mfcc = mfcc.transpose(-1, -2)
    mfcc = fix_length(mfcc, math.ceil(x.shape[-1] / hop_size), -1)
    
    return mfcc


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def pitch(
    x: torch.Tensor,
    sample_rate: int,
    hop_size: int,
    method: Literal["dio", "harvest"] = "harvest",
) -> torch.Tensor:
    """
    Compute the pitch (fundamental frequency) of an audio signal using the specified pitch extraction method.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the audio signal.
        hop_size (int): Hop size for pitch extraction.
        method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        torch.Tensor: Estimated f0 sequence with shape (B?, 1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    device = x.device
    x_np = to_numpy(x, np.float64)
    f0 = core.world.pitch(
        x_np, 
        sample_rate=sample_rate, 
        hop_size=hop_size, 
        method=method
    )
    f0 = from_numpy(f0, device=device, torch_dtype=torch.float32)
    f0 = f0.unsqueeze(1)
    f0 = fix_length(f0, math.ceil(x.shape[-1] / hop_size), -1)
    
    return f0

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def spectral_envelope(
    x: torch.Tensor,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    pitch_extract_method: Literal["dio", "harvest"] = "harvest",
) -> torch.Tensor:
    """
    Compute the spectral envelope of an audio signal using the specified pitch and analysis parameters.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        pitch_extract_method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        torch.Tensor: Spectral envelope with shape (B?, fft_size//2+1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    device = x.device
    
    x_np = to_numpy(x, np.float64)
    sp = core.world.spectral_envelope(
        x_np, 
        sample_rate=sample_rate, 
        fft_size=fft_size, 
        hop_size=hop_size, 
        pitch_extract_method=pitch_extract_method
    )
    sp = from_numpy(sp, device=device, torch_dtype=torch.float32)
    sp = fix_length(sp, math.ceil(x.shape[-1] / hop_size), -1)
    return sp

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def aperiodicity(
    x: torch.Tensor,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    pitch_extract_method: Literal["dio", "harvest"] = "harvest",
) -> torch.Tensor:
    """
    Compute the aperiodicity of an audio signal using the specified pitch and analysis parameters.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        torch.Tensor: Apriodicity with shape (B?, fft_size//2+1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    device = x.device

    x_np = to_numpy(x, np.float64)
    ap = core.world.aperiodicity(
        x_np,
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_size=hop_size,
        pitch_extract_method=pitch_extract_method
    )
    ap = from_numpy(ap, device=device, torch_dtype=torch.float32)
    ap = fix_length(ap, math.ceil(x.shape[-1] / hop_size), -1)
    return ap

 # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def band_aperiodicity(
    x: torch.Tensor,
    sample_rate: int,
    fft_size: int,
    hop_size: int,
    pitch_extract_method: Literal["dio", "harvest"] = "harvest",
) -> torch.Tensor:
    """
    Compute the band aperiodicity of an audio signal using the specified pitch and analysis parameters.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        pitch_extract_method (Literal["dio", "harvest"], optional): Method used for pitch extraction. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        torch.Tensor: Band aperiodicity with shape (B?, N, T)
            - B? is the same as the input.
            - N is the number of bands
            - T is the number of time frames.
    """
    device = x.device

    x_np = to_numpy(x, np.float64)
    bap = core.world.band_aperiodicity(
        x_np,
        sample_rate=sample_rate,
        fft_size=fft_size,
        hop_size=hop_size,
        pitch_extract_method=pitch_extract_method
    )
    bap = from_numpy(bap, device=device, torch_dtype=torch.float32)
    bap = fix_length(bap, math.ceil(x.shape[-1] / hop_size), -1)
    
    return bap

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def vuv(
    x: torch.Tensor,
    sample_rate: int,
    hop_size: int,
    pitch_extract_method: Literal["dio", "harvest"] = "harvest",
) -> torch.Tensor:
    """
    Compute the Voiced/Unvoiced (VUV) mask of an audio signal based on the pitch contour.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sampling rate of the audio signal.
        hop_size (int): Hop size for pitch extraction.
        method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        torch.Tensor: Voice/Unvoiced mask with shape (B?, 1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    f0 = pitch(
        x,
        sample_rate=sample_rate,
        hop_size=hop_size,
        method=pitch_extract_method,
    )
    vuv = (f0 != 0).float()
    return vuv
