import zlib
from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor


def _mask_to_bytes(mask: List[bool]):
    values = [int(i) for i in mask]

    print('sparsity', 1 - sum(values) / len(values))
    while len(values) % 8 != 0:
        values.append(0)
    tokens = []
    for start_i in range(0, len(values), 8):
        token = values[start_i]
        for i in range(1, 8):
            token = (token << 1) + values[start_i + i]
        tokens.append(token)
    return bytes(tokens)


def _bytes_to_mask(mask_bytes: bytes):
    values = []
    for token in mask_bytes:
        tmp = []
        for i in range(8):
            m = token % 2
            tmp.append(m)
            token = token // 2
        for m in reversed(tmp):
            values.append(m)

    print(sum(values) / len(values))
    return values


def encode_mask(mask: Tensor, level=9):
    mask_bytes = _mask_to_bytes([bool(i) for i in mask])
    compressed_mask = zlib.compress(mask_bytes, level=level)  # 注意：这儿要以字节的形式传入
    return compressed_mask


def decode_mask(compressed):
    mask_bytes = zlib.decompress(compressed)
    mask = _bytes_to_mask(mask_bytes)
    return torch.Tensor(mask).long()

