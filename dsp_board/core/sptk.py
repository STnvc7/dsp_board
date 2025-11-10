import numpy as np
import pysptk
from dsp_board.utils import parallelize

@parallelize
def mel_cepstrum(sp: np.ndarray, order: int, alpha: float):
    """
    Compute Mel cepstrum from batched spectral envelope.
    Args:
        sp (np.ndarray): Batched spectral power with shape (B, N, T).
        - B is an optional batch dimension from the input.
        - N is the number of frequency bins.
        - T is the number of time frames.
        order (int): Order of the cepstrum.
        alpha (float): All-pass constant.

    Returns:
        np.ndarray: Mel cepstrum with shape (B, order+1, T).
    """
    mcep = pysptk.sp2mc(sp, order=order, alpha=alpha)
    return mcep
    
@parallelize
def mel_generalized_cepstrum(frames: np.ndarray, order: int, alpha: float, gamma: float):
    """
    Compute Mel generalized cepstrum from batched windowed frames.
    Args:
        frames (np.ndarray): Batched windowed frames with shape (B, N, T).
        order (int): Order of the cepstrum.
        alpha (float): All-pass constant.
        gamma (float): Gamma parameter.

    Returns:
        np.ndarray: Mel generalized cepstrum with shape (B, order+1, T).
    """
    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    return mgc