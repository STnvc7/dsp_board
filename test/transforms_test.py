from typing import Callable, Literal
import pytest

import torch
from dsp_board import Processor
from dsp_board.utils import fix_length

@pytest.mark.torch
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_stft(
    audio_tensor: Callable[[int, str], torch.Tensor], 
    dsp_processor: Processor, 
    dim: int,
    device: Literal["cpu", "cuda"],
):
    if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA is not available on this machine")
            
    x = audio_tensor(dim, device)
    spc = dsp_processor.stft(x)
    inv_x = dsp_processor.istft(spc)
    inv_x = fix_length(inv_x, x.shape[-1], -1)
    
    assert torch.nn.functional.l1_loss(x, inv_x) < 1e-4
    for i in range(inv_x.ndim - 1):
        assert inv_x.shape[i] == x.shape[i]