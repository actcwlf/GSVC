import inspect
import re
import torch
from torch import nn
# from compressai.layers import GDN

torch.set_printoptions(8)

def check_tensor(value):
    stacks = inspect.stack()
    function_name = stacks[0].function
    s = inspect.stack()[1].code_context[0]
    arg_name = re.findall(f'.*{function_name}\((.*?)\)', s)
    nan_count = torch.isnan(value).sum()
    max_weight = value.max()
    min_weight = value.min()
    mean_weight = value.double().mean()
    std_weight = value.double().std()
    print(arg_name,
          '\n    [SHAPE]', value.size(),
          '\n    [DTYPE]', value.dtype,
          '\n    [MEAN]', mean_weight,
          '\n    [STD]', std_weight,
          '\n    [NaN]', nan_count,
          '\n    [MAX]', max_weight,
          '\n    [MIN]', min_weight,
          )



class InspectedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = super().forward(input)
        # check_tensor(output)
        return output


class InspectedConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, output_size=None):
        output = super().forward(input, output_size)
        # check_tensor(output)
        return output

# class InspectedGDN(GDN):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def forward(self, x):
#         output = super().forward(x)
#         # check_tensor(output)
#         return output




def inspectd_conv(in_channels, out_channels, kernel_size=5, stride=2):
    return InspectedConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def inspectd_deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return InspectedConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )