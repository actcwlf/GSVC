import torch
import constriction
import numpy as np
import torch
import os

from typing import Optional, Tuple
from torch import Tensor
import sys

p = os.path.dirname(os.path.abspath(__file__)) + '/../submodules'

sys.path.append(p)

import ans_binding


class ANSCoder:
    def __init__(self, min_symbol, max_symbol, precision=16):
        self.coder = ans_binding.CudaAnsCoder(min_symbol, max_symbol, precision)
        if precision <= 16:
            self.dtype = np.uint16
        elif precision <= 32:
            self.dtype = np.uint32
        else:
            raise ValueError('precision exceeds 32')

    def encode(self, file: str, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor):
        # r_msg = np.array(list(reversed(x.view(-1).cpu())), dtype=np.int32)
        # r_mu = np.array(list(reversed(mu.view(-1).cpu())), dtype=np.float32)
        # r_sigma = np.array(list(reversed(sigma.view(-1).cpu())), dtype=np.float32)

        r_msg = torch.flip(x.view(-1), dims=(0,)).cpu().numpy().astype(dtype=np.int32)
        r_mu = torch.flip(mu.view(-1), dims=(0,)).cpu().numpy().astype(dtype=np.float32)
        r_sigma = torch.flip(sigma.view(-1), dims=(0,)).cpu().numpy().astype(dtype=np.float32)

        compressed: np.ndarray = self.coder.encode_reverse(
            r_msg, r_mu, r_sigma

        )

        compressed = compressed.astype(self.dtype)

        compressed.tofile(file)
        return compressed

    def decode(self, file: str, mu: torch.Tensor, sigma: torch.Tensor):
        compressed = np.fromfile(file, dtype=self.dtype)
        mu = mu.view(-1).cpu().numpy().astype(np.float32)
        sigma = sigma.view(-1).cpu().numpy().astype(np.float32)

        decoded_x = self.coder.decode_v2(compressed.astype(np.uint32), mu, sigma)

        return torch.Tensor(decoded_x)
