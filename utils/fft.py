import torch
# import cv2
import numpy as np

try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft


    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))


    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))


    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)


    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)


    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(x, norm=None):
    X1 = idct(x.transpose(-1, -2), norm=norm)
    X2 = idct(X1.transpose(-1, -2), norm=norm)
    return X2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bgr_img = np.random.random((24, ))
    # bgr_tensor = torch.tensor(bgr_img)
    #
    # dct_cv2 = cv2.dct(bgr_img)
    #
    # print(dct_cv2)
    #
    # # x = torch.rand((5, 10))
    # #
    # x_fft = dct(bgr_tensor.reshape(1, 1, -1), norm='ortho')
    # print(x_fft)
    #
    #
    # img2 = torch.randn((1, 1, 24, 36))
    # img2 = torch.ones((1, 1, 24, 36))
    #
    # plt.imshow(img2.squeeze(0).squeeze(0))
    # plt.show()
    # f_img = dct_2d(img2)
    # plt.imshow(f_img.squeeze(0).squeeze(0))
    # plt.show()

    # print((img2 - idct_2d(dct_2d(img2))).abs().sum())


    f_img = torch.zeros((1, 1, 8, 8))
    f_img[0, 0, -1, -1] = 2
    plt.imshow(f_img.squeeze(0).squeeze(0))
    plt.show()

    r_img = idct_2d(f_img)
    plt.imshow(r_img.squeeze(0).squeeze(0))
    plt.colorbar()
    plt.show()





