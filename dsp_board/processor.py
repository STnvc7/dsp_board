from typing import Literal, Optional
import torch

from dsp_board import transforms
from dsp_board import features
from dsp_board import preprocesses

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class Processor:
    def __init__(
        self,
        sample_rate: int,
        fft_size: int,
        hop_size: int,
        window_size: Optional[int]=None,
        n_mels: int=80,
        n_mfcc: int=13,
        n_mcep: int=39,
        n_mgc: int=24,
        eps: float=1e-8,
        pitch_extract_method: Literal["dio", "harvest"] = "harvest",
        trim_direction: Literal["forward", "backward", "both"] = "forward",
        trim_trigger_level: float=7.0,
        trim_trigger_time: float=0.25,
        target_lufs: float=-24.0,
        target_db: float=-6.0,
    ):
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = window_size if window_size is not None else fft_size
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_mcep = n_mcep
        self.n_mgc = n_mgc
        self.eps = eps
        self.pitch_extract_method: Literal["dio", "harvest"] = pitch_extract_method
        self.trim_direction: Literal["forward", "backward", "both"] = trim_direction
        self.trim_trigger_level = trim_trigger_level
        self.trim_trigger_time = trim_trigger_time
        self.target_lufs = target_lufs
        self.target_db = target_db

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        return transforms.stft(
            x,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            window_size=self.window_size
        )
    def istft(self, x: torch.Tensor) -> torch.Tensor:
        return transforms.istft(
            x,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            window_size=self.window_size
        )
    def spectrogram(self, x: torch.Tensor, power=False, log=True) -> torch.Tensor:
        return features.spectrogram(
            x,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            window_size=self.window_size,
            power=power,
            log=log,
            eps=self.eps
        )
    def mel_spectrogram(self, x):
        return features.mel_spectrogram(
            x,
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            n_mels=self.n_mels,
            window_size=self.window_size,
            log=True,
            eps=self.eps
        )
    def mel_cepstrum(self, x):
        return features.mel_cepstrum(
            x,
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            order=self.n_mcep,
            pitch_extract_method=self.pitch_extract_method,
        )
    def mel_generalized_cepstrum(self, x):
        return features.mel_generalized_cepstrum(
            x,
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            order=self.n_mgc,
        )
    def mfcc(self, x: torch.Tensor) -> torch.Tensor:
        return features.mfcc(
            x,
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc,
            eps=self.eps
        )
    def pitch(self, x: torch.Tensor) -> torch.Tensor:
        return features.pitch(
            x,
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            method=self.pitch_extract_method,
        )
    def spectral_envelope(self, x: torch.Tensor) -> torch.Tensor:
        return features.spectral_envelope(
            x,
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            pitch_extract_method=self.pitch_extract_method,
        )
    def aperiodicity(self, x: torch.Tensor) -> torch.Tensor:
        return features.aperiodicity(
            x,
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            pitch_extract_method=self.pitch_extract_method,
        )
    def band_aperiodicity(self, x: torch.Tensor) -> torch.Tensor:
        return features.band_aperiodicity(
            x,
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            hop_size=self.hop_size,
            pitch_extract_method=self.pitch_extract_method,
        )
    def vuv(self, x: torch.Tensor) -> torch.Tensor:
        return features.vuv(
            x,
            sample_rate=self.sample_rate,
            hop_size=self.hop_size,
            pitch_extract_method=self.pitch_extract_method,
        )
    def resample(self, x: torch.Tensor, target_sample_rate: int) -> torch.Tensor:
        return preprocesses.resample(
            x,
            original_sample_rate=self.sample_rate,
            target_sample_rate=target_sample_rate
        )
    def trim(self, x: torch.Tensor, direction: Optional[Literal["forward", "backward", "both"]]=None) -> torch.Tensor:
        return preprocesses.trim(
            x,
            sample_rate=self.sample_rate,
            direction=self.trim_direction if direction is None else direction,
            trigger_level=self.trim_trigger_level,
            trigger_time=self.trim_trigger_time
        )
    def loudness_normalize(self, x: torch.Tensor, target_lufs: Optional[float]=None) -> torch.Tensor:
        return preprocesses.loudness_normalize(
            x,
            sample_rate=self.sample_rate,
            target_lufs=self.target_lufs if target_lufs is None else target_lufs
        )
    def peak_normalize(self, x: torch.Tensor, target_db: Optional[float]=None) -> torch.Tensor:
        return preprocesses.peak_normalize(
            x,
            target_db=self.target_db if target_db is None else target_db
        )
