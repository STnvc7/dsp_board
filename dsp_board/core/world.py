from typing import Literal
import numpy as np
import pyworld as pw

from dsp_board.utils.parallelize import parallelize

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@parallelize
def pitch(
    x: np.ndarray, 
    sample_rate: int, 
    hop_size: int, 
    method: Literal["dio", "harvest"]="harvest"
) -> np.ndarray:
    """
    Compute pitch from batched waveform.
    Args:
        x (np.ndarray): Input waveform with shape (B, L).
        sample_rate (int): Sampling rate of the input signal.
        hop_size (int): Number of samples between successive frames (frame shift).
        method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        np.ndarray: f0 frequency with shape (B, T)
    """
    if method == "dio":
        f0, t = pw.dio(x, sample_rate, frame_period=hop_size/sample_rate * 1000) # pyright: ignore[reportAttributeAccessIssue]
        f0 = pw.stonemask(x, f0, t, sample_rate) # pyright: ignore[reportAttributeAccessIssue]
    elif method == "harvest":
        f0, t = pw.harvest(x, sample_rate, frame_period=hop_size/sample_rate * 1000) # pyright: ignore[reportAttributeAccessIssue]
    else:
        raise ValueError(f"Invalid method: {method}")
    return f0
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@parallelize
def spectral_envelope(
    x: np.ndarray, 
    sample_rate: int, 
    fft_size: int, 
    hop_size: int, 
    pitch_extract_method: Literal["dio", "harvest"]="harvest"
) -> np.ndarray:
    """
    Compute spectral envelope from batched waveform and estimated pitch values.
    Args:
        x (np.ndarray): Input waveform with shape (B, L).
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        pitch_extract_method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        np.ndarray: Spectral envelope with shape (B, fft_size//2+1, T).
    """
    if pitch_extract_method == "dio":
        f0, t = pw.dio(x, sample_rate, frame_period=hop_size/sample_rate * 1000) # pyright: ignore[reportAttributeAccessIssue]
        f0 = pw.stonemask(x, f0, t, sample_rate) # pyright: ignore[reportAttributeAccessIssue]
    elif pitch_extract_method == "harvest":
        f0, t = pw.harvest(x, sample_rate, frame_period=hop_size/sample_rate * 1000) # pyright: ignore[reportAttributeAccessIssue]
    else:
        raise ValueError(f"Invalid method: {pitch_extract_method}")
        
    sp = pw.cheaptrick(x, f0, t, fs=sample_rate, fft_size=fft_size) # pyright: ignore[reportAttributeAccessIssue]
    sp = sp.transpose(1,0)
    return sp
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@parallelize
def aperiodicity(
    x: np.ndarray, 
    sample_rate: int, 
    fft_size: int, 
    hop_size: int,
    pitch_extract_method: Literal["dio", "harvest"]="harvest"
) -> np.ndarray:
    """
    Compute aperiodicity from batched waveform.
    Args:
        x (np.ndarray): Input waveform with shape (B, L).
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        pitch_extract_method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        np.ndarray: Aperiodicity with shape (B, fft_size//2+1, T).
    """
    if pitch_extract_method == "dio":
        f0, t = pw.dio(x, sample_rate, frame_period=hop_size/sample_rate * 1000) # pyright: ignore[reportAttributeAccessIssue]
        f0 = pw.stonemask(x, f0, t, sample_rate) # pyright: ignore[reportAttributeAccessIssue]
    elif pitch_extract_method == "harvest":
        f0, t = pw.harvest(x, sample_rate, frame_period=hop_size/sample_rate * 1000) # pyright: ignore[reportAttributeAccessIssue]
    else:
        raise ValueError(f"Invalid method: {pitch_extract_method}")
        
    ap = pw.d4c(x, f0, t, fs=sample_rate, fft_size=fft_size) # pyright: ignore[reportAttributeAccessIssue]
    ap = ap.transpose(1,0)
    return ap

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@parallelize
def band_aperiodicity(
    x: np.ndarray, 
    sample_rate: int, 
    fft_size: int, 
    hop_size: int,
    pitch_extract_method: Literal["dio", "harvest"]="harvest"
) -> np.ndarray:
    """
    Compute band aperiodicity from batched waveform.
    Args:
        x (np.ndarray): Input waveform with shape (B, L).
        sample_rate (int): Sampling rate of the input signal.
        fft_size (int): FFT window size for computing the STFT.
        hop_size (int): Number of samples between successive frames (frame shift).
        pitch_extract_method (Literal["dio", "harvest"], optional): The pitch extraction method to use. Options are "dio" or "harvest". Defaults to "harvest".

    Returns:
        np.ndarray: Band aperiodicity with shape (B, N, L).
    """
    if pitch_extract_method == "dio":
        f0, t = pw.dio(x, sample_rate, frame_period=hop_size/sample_rate * 1000) # pyright: ignore[reportAttributeAccessIssue]
        f0 = pw.stonemask(x, f0, t, sample_rate) # pyright: ignore[reportAttributeAccessIssue]
    elif pitch_extract_method == "harvest":
        f0, t = pw.harvest(x, sample_rate, frame_period=hop_size/sample_rate * 1000) # pyright: ignore[reportAttributeAccessIssue]
    else:
        raise ValueError(f"Invalid method: {pitch_extract_method}")
        
    ap = pw.d4c(x, f0, t, fs=sample_rate, fft_size=fft_size) # pyright: ignore[reportAttributeAccessIssue]
    bap = pw.code_aperiodicity(ap, sample_rate) # pyright: ignore[reportAttributeAccessIssue]
    bap = bap.transpose(1,0)
    return bap