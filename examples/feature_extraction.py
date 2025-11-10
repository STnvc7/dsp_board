import os
import sys

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)

import time

import torch
import torchaudio

from dsp_board import set_max_workers
from dsp_board.features import mel_cepstrum

sample_rate = 48000
fft_size = 1024
hop_size = 256

x = [torchaudio.load(f"./samples/BASIC5000_000{i + 1}.wav")[0] for i in range(0, 8)]
offset = min([_x.shape[-1] for _x in x])
x = torch.stack([_x[..., :offset] for _x in x])
print(x.shape)

mcep_unbatch = mel_cepstrum(x[0], sample_rate, fft_size, hop_size, 20)
print(mcep_unbatch.shape)

set_max_workers(1)
start = time.perf_counter()
mcep = mel_cepstrum(x, sample_rate, fft_size, hop_size, 20)
end = time.perf_counter()
print(f"1 worker: {end - start}s")


set_max_workers(12)
start = time.perf_counter()
mcep_batch = mel_cepstrum(x, sample_rate, fft_size, hop_size, 20)
end = time.perf_counter()
print(f"12 workers: {end - start}s")
print(mcep_batch.shape)
