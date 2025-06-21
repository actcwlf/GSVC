import typing
from dataclasses import dataclass

import torch

# from ortho_gaussian_renderer import GeneratedGaussians


@dataclass
class RenderResults:
    rendered_image: torch.Tensor
    viewspace_points: torch.Tensor
    visible_mask: torch.Tensor
    visibility_filter: torch.Tensor
    radii: torch.Tensor
    active_gaussains: int
    num_rendered: int
    time_sub: typing.Union[torch.Tensor, None] = None
    selection_mask: typing.Union[torch.Tensor, None] = None
    neural_opacity: typing.Union[torch.Tensor, None] = None
    scaling: typing.Union[torch.Tensor, None] = None
    bit_per_param: typing.Union[torch.Tensor, None] = None
    bit_per_feat_param: typing.Union[torch.Tensor, None] = None
    bit_per_scaling_param: typing.Union[torch.Tensor, None] = None
    bit_per_offsets_param: typing.Union[torch.Tensor, None] = None
    generated_gaussians: typing.Any = None
    entropy_constrained: bool = False
