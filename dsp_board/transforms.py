from typing import Optional
import torch
import torch.nn.functional as F
import math
from dsp_board.utils import channelize, fix_length

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def stft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    window_size: Optional[int]=None,
) -> torch.Tensor:
    """
    Compute the complex spectrogram from an input waveform using STFT.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        fft_size (int): Number of FFT points (i.e., window length for FFT).
        hop_size (int): Number of audio samples between adjacent STFT columns (frame shift).
        window_size (Optional[int], optional): Length of the window function. If None, defaults to fft_size.

    Returns:
        torch.Tensor: Complex spectrogram with shape (B?, fft_size//2+1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """
    pad_size = (fft_size - hop_size) // 2
    x_padded = F.pad(x, (pad_size, pad_size))

    window_size = fft_size if window_size is None else window_size
    window = torch.hann_window(window_size)

    spc = torch.stft(
        x_padded,
        n_fft=fft_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window.to(x.device),
        center=False,
        onesided=True,
        normalized=True,
        return_complex=True,
    )

    spc = fix_length(spc, math.ceil(x.shape[-1]/hop_size), -1)
    return spc

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=2)
def istft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    window_size: Optional[int]=None
) -> torch.Tensor:
    """
    Compute the inverse STFT to reconstruct the waveform from a complex spectrogram.

    Args:
        x (torch.Tensor): Input complex spectrogram of shape (B?, fft_size//2+1, T).
            - B? is an optional batch dimension from the input.
            - T is the number of time frames.
        fft_size (int): Number of FFT points (i.e., window length for FFT).
        hop_size (int): Number of audio samples between adjacent STFT columns (frame shift).
        window_size (Optional[int], optional): Length of the window function. If None, defaults to fft_size.

    Returns:
        torch.Tensor: Reconstructed waveform with shape. (B?, 1, T)
            - B? is the same as the input.
            - T is the number of time frames.
    """

    window_size = fft_size if window_size is None else window_size
    window = torch.hann_window(window_size, device=x.device)
    pad_size = (window_size - hop_size) // 2

    B, N, T = x.shape
    # Inverse FFT
    ifft = torch.fft.irfft(x, fft_size, dim=1, norm="ortho")
    ifft = ifft * window[None, :, None]

    # Overlap and Add
    output_size = (T - 1) * hop_size + window_size
    y = torch.nn.functional.fold(
        ifft,
        output_size=(1, output_size),
        kernel_size=(1, window_size),
        stride=(1, hop_size),
    )[:, 0, 0, pad_size:-pad_size]

    # Window envelope
    window_sq = window.square().expand(1, T, -1).transpose(1, 2)
    window_envelope = torch.nn.functional.fold(
        window_sq,
        output_size=(1, output_size),
        kernel_size=(1, window_size),
        stride=(1, hop_size),
    ).squeeze()[pad_size:-pad_size]

    y = y / window_envelope

    y = fix_length(y, math.ceil(x.shape[-1]*hop_size))
    
    return y
