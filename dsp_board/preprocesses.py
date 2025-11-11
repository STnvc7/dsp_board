from typing import Literal
import torch
import torchaudio

from dsp_board.utils import channelize

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def resample(
    x: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int
) -> torch.Tensor:
    """
    Resample an audio signal to a target sample rate.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        original_sample_rate (int): Original sample rate of the audio signal.
        target_sample_rate (int): Desired target sample rate.

    Returns:
        torch.Tensor: Resampled waveform with shape (B?, L').
            - B? is the same as the input.
            - L' is the length of the resampled waveform.
    """
    x = torchaudio.functional.resample(
        x,
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
        resampling_method="sinc_interp_hann",
    )
    return x

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def trim(
    x: torch.Tensor,
    sample_rate: int,
    direction: Literal["forward", "backward", "both"] = "forward",
    trigger_level: float = 7.0,
    trigger_time: float = 0.25
):
    """
    Trim leading and trailing silence from an audio signal using VAD.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sample rate of the audio signal.
        direction (Literal["forward", "backward", "both"]): Direction of trimming. Defaults to "forward".
        trigger_level (Optional[float], optional): VAD trigger level (sensitivity).
            Higher values are stricter. Defaults to 7.0.
        trigger_time (Optional[float], optional): Duration (in seconds) of audio required
            to trigger VAD. Defaults to 0.25.

    Returns:
        torch.Tensor: Audio signal with leading and trailing silence removed with shape (B?, L').
            - B? is the same as the input.
            - L is the length of the trimmed waveform.
    """
    if direction == "forward":
        trimmed = torchaudio.functional.vad(
            x, sample_rate, trigger_level=trigger_level, trigger_time=trigger_time
        )

    elif direction == "backward":
        back_trimmed = torchaudio.functional.vad(
            x.flip(dims=[-1]), sample_rate, trigger_level=trigger_level, trigger_time=trigger_time,
        )
        trimmed = back_trimmed.flip(dims=[-1])
    elif direction == "both":
        front_trimmed = torchaudio.functional.vad(
            x, sample_rate, trigger_level=trigger_level, trigger_time=trigger_time
        )
        back_trimmed = torchaudio.functional.vad(
            front_trimmed.flip(dims=[-1]), sample_rate, trigger_level=trigger_level, trigger_time=trigger_time,
        )
        trimmed = back_trimmed.flip(dims=[-1])
    else:
        raise ValueError(f"Invalid direction: {direction}")

    return trimmed

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def loudness_normalize(
    x: torch.Tensor,
    sample_rate: int,
    target_lufs: float = -24.0,
):
    """
    Normalize the loudness of an audio signal.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        sample_rate (int): Sample rate of the audio signal.
        target_lufs (float, optional): Target loudness level in LUFS. Defaults to -24.0.

    Returns:
        torch.Tensor: Waveform with normalized loudness with shape (B?, L).
    """
    normalized = []
    for i in range(x.shape[0]):
        _x = x[i].unsqueeze(0)
        input_lufs = torchaudio.functional.loudness(_x, sample_rate)
        delta_lufs = target_lufs - input_lufs
        gain = torch.pow(10.0, delta_lufs / 20.0)
        _x = _x * gain
        normalized += [_x.squeeze(0)]
        
    normalized = torch.stack(normalized, dim=0)
    return normalized

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@channelize(keep_dims=1)
def peak_normalize(
    x: torch.Tensor,
    target_db: float = -1.0
):
    """
    Normalize the peak amplitude of an audio signal.

    Args:
        x (torch.Tensor): Input waveform of shape (B?, L).
            - B? is an optional batch dimension from the input.
            - L is the length of the input waveform.
        target_db (float, optional): Target peak amplitude in dBFS. Defaults to -6.0.

    Returns:
        torch.Tensor: Waveform with normalized peak amplitude with shape (B?, L).
    """
    peak = x.abs().max(dim=-1, keepdim=True).values
    peak_db = 20.0 * torch.log10(peak + 1e-12)
    gain_db = target_db - peak_db
    gain = torch.pow(10.0, gain_db / 20.0)
    normalized = x * gain
    return normalized
