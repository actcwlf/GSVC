import torch


def quantize_per_tensor(t, bit=8, eps=1e-19):
    valid_mask = (t != 0)
    t_min, t_max = float(t[valid_mask].min()), float(t[valid_mask].max())
    scale = (t_max - t_min) / 2 ** bit

    quant_t = ((t - t_min) / (scale + eps)).round()
    rescaled_t = t_min + scale * quant_t

    new_t = torch.zeros_like(t)
    new_t[valid_mask] = rescaled_t[valid_mask]

    meta_info = {
        't_min': t_min,
        'scale': scale
    }

    return quant_t, valid_mask, new_t, meta_info


def quantize_per_dimension(t, bit=8, axis=0, eps=1e-19):
    assert axis == 0, "Only quantization along first dimension is supported for now"
    quant_t_list = []
    valid_mask_list = []
    new_t_list = []
    meta_info_list = []
    for i in range(t.size(axis)):
        quant_t, valid_mask, new_t, meta_info = quantize_per_tensor(t[i:i+1, ...], bit=bit, eps=eps)
        quant_t_list.append(quant_t)
        valid_mask_list.append(valid_mask)
        new_t_list.append(new_t)
        meta_info_list.append(meta_info)

    quant_t = torch.cat(quant_t_list, dim=0)
    valid_mask = torch.cat(valid_mask_list, dim=0)
    new_t = torch.cat(new_t_list, dim=0)
    meta_info = {
        't_min': [],
        'scale': []
    }
    for m in meta_info_list:
        meta_info['t_min'].append(m['t_min'])
        meta_info['scale'].append(m['scale'])
    return quant_t, valid_mask, new_t, meta_info


def quantize_tensor(t, bit=16, axis=-1, eps=1e-19):
    if axis == -1:
        return quantize_per_tensor(t, bit, eps)
    else:
        return quantize_per_dimension(t, bit, axis, eps)


def dequantize_tensor(quant_t, mask, meta):
    new_dims = len(quant_t.shape) - 1
    t_min = torch.tensor(meta['t_min']).view(-1, *([1] * new_dims))
    scale = torch.tensor(meta['scale']).view(-1, *([1] * new_dims))
    rescaled_t = t_min + scale * quant_t
    return rescaled_t * mask