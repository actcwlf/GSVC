import torch
import constriction
import numpy as np
import torch

from typing import Optional, Tuple
from torch import Tensor

# from utils.constants import Q_PROBA_DEFAULT
Q_PROBA_DEFAULT = 128.0



class RangeCoder:
    def __init__(self, min_symbol: int, max_symbol: int, Q_PROBA: int = Q_PROBA_DEFAULT):

        # Higher: more accurate but less reliable probability model
        # Actual q_step is 1 / Q_PROBA
        self.Q_PROBA = 1024.0 # Q_PROBA

        # Data are in [-AC_MAX_VAL, AC_MAX_VAL - 1]
        # self.AC_MAX_VAL = AC_MAX_VAL

        # self.alphabet = np.arange(-self.AC_MAX_VAL, self.AC_MAX_VAL + 1)
        self.model_family = constriction.stream.model.QuantizedGaussian(
            min_symbol, max_symbol
        )

        # self.n_ctx_rowcol = n_ctx_rowcol

    def quantize_proba_parameters(self, x: Tensor, mode: str = 'mu') -> Tensor:
        """Apply a quantization to the input x to reduce floating point
        drift.

        Args:
            x (Tensor): The value to quantize

        Returns:
            Tensor: the quantize value
        """

        # return x
        return torch.round(x * self.Q_PROBA) / self.Q_PROBA + 1e-6


    def encode(
        self,
        out_file: str,
        x: Tensor,
        mu: Tensor,
        scale: Tensor,
    ):
        """Encode a 1D tensor x, using two 1D tensors mu and scale for the
        element-wise probability model of x.

        Args:
            x (Tensor): [B] tensor of values to be encoded
            mu (Tensor): [B] tensor describing the expectation of x
            scale (Tensor): [B] tensor with the standard deviations of x
        """


        mu = self.quantize_proba_parameters(mu, mode = 'mu')
        scale = self.quantize_proba_parameters(scale, mode = 'scale')

        # proba = laplace_cdf(x + 0.5, mu, scale) - laplace_cdf(x - 0.5, mu, scale)
        # entropy_rate_bit = -torch.log2(torch.clamp_min(proba, min = 2 ** -16)).sum()

        # x = torch.flip(x, dims=(0, )).numpy().astype(np.int32)
        # mu = torch.flip(mu, dims=(0, )).numpy().astype(np.float64)
        # scale = torch.flip(scale, dims=(0, )).numpy().astype(np.float64)

        x = x.numpy().astype(np.int32)
        mu = mu.numpy().astype(np.float64)
        scale = scale.numpy().astype(np.float64)

        # encoder = constriction.stream.queue.RangeEncoder()
        # encoder.encode(x, self.model_family, mu, scale)

        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(x, self.model_family, mu, scale)
        # encoder.get_compressed().tofile(out_file)

        with open(out_file, 'wb') as f_out:
            f_out.write(encoder.get_compressed())

    def load_bitstream(self, in_file: str):
        bitstream = np.fromfile(in_file, dtype=np.uint32)
        # self.decoder = constriction.stream.queue.RangeDecoder(bitstream)
        self.decoder = constriction.stream.stack.AnsCoder(bitstream)


    def decode(self, mu: Tensor, scale: Tensor) -> Tensor:

        mu = self.quantize_proba_parameters(mu)
        scale = self.quantize_proba_parameters(scale)

        mu = mu.numpy().astype(np.float64)
        scale = scale.numpy().astype(np.float64)
        # mu = torch.flip(mu, dims=(0, )).numpy().astype(np.float64)
        # scale = torch.flip(scale, dims=(0, )).numpy().astype(np.float64)
        x = self.decoder.decode(self.model_family, mu, scale)

        x = torch.tensor(x).to(torch.float)

        return x


