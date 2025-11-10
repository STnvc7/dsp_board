from dsp_board import features
from dsp_board import set_enable_parallel, set_max_workers
import math
import pytest

class TestFeatures:
    @pytest.mark.torch
    def test_spectrogram_1d(self, audio_1d, dsp_config):
        spc = features.spectrogram(
            audio_1d,
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            window_size=dsp_config["window_size"],
            power=False,
            log=True
        )
        assert spc.shape[0] == dsp_config["fft_size"] // 2 + 1
        assert spc.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])
    
    @pytest.mark.torch
    def test_spectrogram_2d(self, audio_2d, dsp_config):
        spc = features.spectrogram(
            audio_2d,
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            window_size=dsp_config["window_size"],
            power=False,
            log=True
        )
        assert spc.shape[0] == audio_2d.shape[0]
        assert spc.shape[1] == dsp_config["fft_size"] // 2 + 1
        assert spc.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.torch
    def test_spectrogram_3d(self, audio_3d, dsp_config):
        spc = features.spectrogram(
            audio_3d,
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            window_size=dsp_config["window_size"],
            power=False,
            log=True
        )
        assert spc.shape[0] == audio_3d.shape[0]
        assert spc.shape[1] == audio_3d.shape[1]
        assert spc.shape[2] == dsp_config["fft_size"] // 2 + 1
        assert spc.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.torch
    def test_mel_spectrogram_1d(self, audio_1d, dsp_config):
        spc = features.mel_spectrogram(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            n_mels=dsp_config["n_mels"],
            window_size=dsp_config["window_size"],
            power=False,
            log=True
        )
        assert spc.shape[0] == dsp_config["n_mels"]
        assert spc.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.torch
    def test_mel_spectrogram_2d(self, audio_2d, dsp_config):
        spc = features.mel_spectrogram(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            n_mels=dsp_config["n_mels"],
            window_size=dsp_config["window_size"],
            power=False,
            log=True
        )
        assert spc.shape[0] == audio_2d.shape[0]
        assert spc.shape[1] == dsp_config["n_mels"]
        assert spc.shape[2] == math.ceil(audio_2d.shape[1] / dsp_config["hop_size"])

    @pytest.mark.torch
    def test_mel_spectrogram_3d(self, audio_3d, dsp_config):
        spc = features.mel_spectrogram(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            n_mels=dsp_config["n_mels"],
            window_size=dsp_config["window_size"],
            power=False,
            log=True
        )
        assert spc.shape[0] == audio_3d.shape[0]
        assert spc.shape[1] == audio_3d.shape[1]
        assert spc.shape[2] == dsp_config["n_mels"]
        assert spc.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.torch
    def test_mfcc_1d(self, audio_1d, dsp_config):
        mfcc = features.mfcc(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            n_mels=dsp_config["n_mels"],
            n_mfcc=dsp_config["n_mfcc"],
        )
        assert mfcc.shape[0] == dsp_config["n_mfcc"]
        assert mfcc.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.torch
    def test_mfcc_2d(self, audio_2d, dsp_config):
        mfcc = features.mfcc(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            n_mels=dsp_config["n_mels"],
            n_mfcc=dsp_config["n_mfcc"],
        )
        assert mfcc.shape[0] == audio_2d.shape[0]
        assert mfcc.shape[1] == dsp_config["n_mfcc"]
        assert mfcc.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.torch
    def test_mfcc_3d(self, audio_3d, dsp_config):
        mfcc = features.mfcc(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            n_mels=dsp_config["n_mels"],
            n_mfcc=dsp_config["n_mfcc"],
        )
        assert mfcc.shape[0] == audio_3d.shape[0]
        assert mfcc.shape[1] == audio_3d.shape[1]
        assert mfcc.shape[2] == dsp_config["n_mfcc"]
        assert mfcc.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.sptk
    def test_mel_cepstrum_1d(self, audio_1d, dsp_config):
        mcep = features.mel_cepstrum(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            order=dsp_config["n_mcep"],
        )
        assert mcep.shape[0] == dsp_config["n_mcep"]+1
        assert mcep.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.sptk
    @pytest.mark.parametrize("enable_parallel, max_workers", [(False, 1), (True, 4), (True, 8)])
    def test_mel_cepstrum_2d(self, audio_2d, dsp_config, enable_parallel, max_workers):
        set_enable_parallel(enable_parallel)
        set_max_workers(max_workers)

        mcep = features.mel_cepstrum(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            order=dsp_config["n_mcep"],
        )
        assert mcep.shape[0] == audio_2d.shape[0]
        assert mcep.shape[1] == dsp_config["n_mcep"]+1
        assert mcep.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.sptk
    def test_mel_cepstrum_3d(self, audio_3d, dsp_config):
        mcep = features.mel_cepstrum(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            order=dsp_config["n_mcep"],
        )
        assert mcep.shape[0] == audio_3d.shape[0]
        assert mcep.shape[1] == audio_3d.shape[1]
        assert mcep.shape[2] == dsp_config["n_mcep"]+1
        assert mcep.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.sptk
    def test_mel_generalized_cepstrum_1d(self, audio_1d, dsp_config):
        mcep = features.mel_generalized_cepstrum(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            order=dsp_config["n_mgc"],
        )
        assert mcep.shape[0] == dsp_config["n_mgc"]+1
        assert mcep.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.sptk
    @pytest.mark.slow
    @pytest.mark.parametrize("enable_parallel, max_workers", [(False, 1), (True, 4), (True, 8)])
    def test_mel_generalized_cepstrum_2d(self, audio_2d, dsp_config, enable_parallel, max_workers):
        set_enable_parallel(enable_parallel)
        set_max_workers(max_workers)

        mcep = features.mel_generalized_cepstrum(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            order=dsp_config["n_mgc"],
        )
        assert mcep.shape[0] == audio_2d.shape[0]
        assert mcep.shape[1] == dsp_config["n_mgc"]+1
        assert mcep.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])
    
    @pytest.mark.sptk
    @pytest.mark.slow
    def test_mel_generalized_cepstrum_3d(self, audio_3d, dsp_config):
        B, C, T = audio_3d.shape
        target_length = math.ceil(T / dsp_config["hop_size"])

        mcep = features.mel_generalized_cepstrum(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
            order=dsp_config["n_mgc"],
        )
        assert mcep.shape[0] == B
        assert mcep.shape[1] == C
        assert mcep.shape[2] == dsp_config["n_mgc"]+1
        assert mcep.shape[3] == target_length

    @pytest.mark.world
    @pytest.mark.parametrize("method", ["dio", "harvest"])
    def test_pitch_1d(self, audio_1d, dsp_config, method):
        pitch = features.pitch(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            hop_size=dsp_config["hop_size"],
            method=method
        )
        assert pitch.shape[0] == 1
        assert pitch.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    @pytest.mark.parametrize("enable_parallel, max_workers", [(False, 1), (True, 4), (True, 8)])
    def test_pitch_2d(self, audio_2d, dsp_config, enable_parallel, max_workers):
        set_enable_parallel(enable_parallel)
        set_max_workers(max_workers)

        pitch = features.pitch(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            hop_size=dsp_config["hop_size"],
        )
        assert pitch.shape[0] == audio_2d.shape[0]
        assert pitch.shape[1] == 1
        assert pitch.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_pitch_3d(self, audio_3d, dsp_config):
        pitch = features.pitch(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            hop_size=dsp_config["hop_size"],
        )
        assert pitch.shape[0] == audio_3d.shape[0]
        assert pitch.shape[1] == audio_3d.shape[1]
        assert pitch.shape[2] == 1
        assert pitch.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_spectral_envelope_1d(self, audio_1d, dsp_config):
        spectral_envelope = features.spectral_envelope(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert spectral_envelope.shape[0] == dsp_config["fft_size"] // 2 + 1
        assert spectral_envelope.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    @pytest.mark.parametrize("enable_parallel, max_workers", [(False, 1), (True, 4), (True, 8)])
    def test_spectral_envelope_2d(self, audio_2d, dsp_config, enable_parallel, max_workers):
        set_enable_parallel(enable_parallel)
        set_max_workers(max_workers)

        spectral_envelope = features.spectral_envelope(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert spectral_envelope.shape[0] == audio_2d.shape[0]
        assert spectral_envelope.shape[1] == dsp_config["fft_size"] // 2 + 1
        assert spectral_envelope.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_spectral_envelope_3d(self, audio_3d, dsp_config):
        spectral_envelope = features.spectral_envelope(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert spectral_envelope.shape[0] == audio_3d.shape[0]
        assert spectral_envelope.shape[1] == audio_3d.shape[1]
        assert spectral_envelope.shape[2] == dsp_config["fft_size"] // 2 + 1
        assert spectral_envelope.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_aperiodicity_1d(self, audio_1d, dsp_config):
        aperiodicity = features.aperiodicity(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert aperiodicity.shape[0] == dsp_config["fft_size"] // 2 + 1
        assert aperiodicity.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    @pytest.mark.parametrize("enable_parallel, max_workers", [(False, 1), (True, 4), (True, 8)])
    def test_aperiodicity_2d(self, audio_2d, dsp_config, enable_parallel, max_workers):
        aperiodicity = features.aperiodicity(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert aperiodicity.shape[0] == audio_2d.shape[0]
        assert aperiodicity.shape[1] == dsp_config["fft_size"] // 2 + 1
        assert aperiodicity.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_aperiodicity_3d(self, audio_3d, dsp_config):
        aperiodicity = features.aperiodicity(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert aperiodicity.shape[0] == audio_3d.shape[0]
        assert aperiodicity.shape[1] == audio_3d.shape[1]
        assert aperiodicity.shape[2] == dsp_config["fft_size"] // 2 + 1
        assert aperiodicity.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_band_aperiodicity_1d(self, audio_1d, dsp_config):
        aperiodicity = features.band_aperiodicity(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert aperiodicity.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    @pytest.mark.parametrize("enable_parallel, max_workers", [(False, 1), (True, 4), (True, 8)])
    def test_band_aperiodicity_2d(self, audio_2d, dsp_config, enable_parallel, max_workers):
        aperiodicity = features.band_aperiodicity(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert aperiodicity.shape[0] == audio_2d.shape[0]
        assert aperiodicity.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_band_aperiodicity_3d(self, audio_3d, dsp_config):
        aperiodicity = features.band_aperiodicity(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            fft_size=dsp_config["fft_size"],
            hop_size=dsp_config["hop_size"],
        )
        assert aperiodicity.shape[0] == audio_3d.shape[0]
        assert aperiodicity.shape[1] == audio_3d.shape[1]
        assert aperiodicity.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_vuv_1d(self, audio_1d, dsp_config):
        vuv = features.vuv(
            audio_1d,
            sample_rate=dsp_config["sample_rate"],
            hop_size=dsp_config["hop_size"],
        )
        assert vuv.shape[0] == 1
        assert vuv.shape[1] == math.ceil(audio_1d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    @pytest.mark.parametrize("enable_parallel, max_workers", [(False, 1), (True, 4), (True, 8)])
    def test_vuv_2d(self, audio_2d, dsp_config, enable_parallel, max_workers):
        set_enable_parallel(enable_parallel)
        set_max_workers(max_workers)

        vuv = features.vuv(
            audio_2d,
            sample_rate=dsp_config["sample_rate"],
            hop_size=dsp_config["hop_size"],
        )
        assert vuv.shape[0] == audio_2d.shape[0]
        assert vuv.shape[1] == 1
        assert vuv.shape[2] == math.ceil(audio_2d.shape[-1] / dsp_config["hop_size"])

    @pytest.mark.world
    def test_vuv_3d(self, audio_3d, dsp_config):
        vuv = features.vuv(
            audio_3d,
            sample_rate=dsp_config["sample_rate"],
            hop_size=dsp_config["hop_size"],
        )
        assert vuv.shape[0] == audio_3d.shape[0]
        assert vuv.shape[1] == audio_3d.shape[1]
        assert vuv.shape[2] == 1
        assert vuv.shape[3] == math.ceil(audio_3d.shape[-1] / dsp_config["hop_size"])