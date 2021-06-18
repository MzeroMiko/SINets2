import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Weighted_loss(nn.Module):
    def __init__(self, s=11, var=5):
        super(Weighted_loss, self).__init__()    
        self.s = s
        self.var = var
        self.eps = 1e-8
        self.weight = self.get_gaussian_weight(self.s, self.var)
        
    def get_gaussian_weight(self, s=11, var=5):
        x = np.tile(np.expand_dims(np.arange(s), axis=0),(s,1)) - (s-1)/2
        y = np.exp(-(x**2 + x.T**2)/(2*var))
        y = y/y.sum() # normalize
        return nn.Parameter(torch.from_numpy(y[np.newaxis, np.newaxis, ...]).float(), requires_grad=False)

    def forward(self, pred, gt):
        gt = gt.type_as(pred)
        z = torch.zeros_like(pred)
        score1 = torch.where(gt > 0, torch.sigmoid(pred), z)
        weight_foreground = 1 - F.conv2d(score1, self.weight.type_as(pred), padding=int(self.s/2))
        weight = torch.where(gt > 0, weight_foreground, 1 - z)
        loss_BCE = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        weight_loss = (weight * loss_BCE).mean()
        
        # pred = torch.sigmoid(pred)
        # inter = (pred*gt).sum(dim=(2,3))
        # union = (pred+gt).sum(dim=(2,3))
        # wiou = 1 - inter /(union - inter + self.eps)
        
        # 目前只是简单相加，之后或许可以分配权重
        # print(weight_loss, wiou.mean())
        # return (weight_loss + wiou.mean())
        return weight_loss


class wiouloss(nn.Module):
    def __init__(self, s=11, var=5):
        super(wiouloss, self).__init__()
        self.eps = 1e-8

    def forward(self, pred, gt):
        pred = torch.sigmoid(pred)
        inter = (pred * gt).sum(dim=(2,3))
        union = (pred + gt).sum(dim=(2,3))
        wiou = 1 - inter / (union - inter + self.eps)        
        return wiou.mean()


def get_lowpass_gaussian_filter(L, D0):
    x = np.tile(np.expand_dims(np.arange(L), axis=0), (L,1)) - (L-1)/2
    y = x.T
    D = x**2 + y**2
    F = np.exp(-D/(2*D0**2))
    
    return nn.Parameter(torch.from_numpy(np.fft.fftshift(F)), requires_grad=False).view(1,1,L,L).expand(1,3,L,L).cuda()


def get_highpass_gaussian_filter(L, D0):
    return 1 - get_lowpass_gaussian_filter(L, D0)


import numpy as np
import torch
import torch.nn as nn

def _rfft(x, signal_ndim=1, onesided=True):
    # b = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
    # b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
    # torch 1.8.0 torch.fft.rfft  to torch 1.5.0 torch.rfft as signal_ndim=1
    # written by mzero
    odd_shape1 = (x.shape[1] % 2 != 0)
    x = torch.fft.rfft(x)
    x = torch.cat([x.real.unsqueeze(dim=2), x.imag.unsqueeze(dim=2)], dim=2)
    if onesided == False:
        _x = x[:, 1:, :].flip(dims=[1]).clone() if odd_shape1 else x[:, 1:-1, :].flip(dims=[1]).clone()
        _x[:,:,1] = -1 * _x[:,:,1]
        x = torch.cat([x, _x], dim=1)
    return x


def _irfft(x, signal_ndim=1, onesided=True):
    # b = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
    # b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
    # torch 1.8.0 torch.fft.irfft  to torch 1.5.0 torch.irfft as signal_ndim=1
    # written by mzero
    if onesided == False:
        res_shape1 = x.shape[1]
        x = x[:,:(x.shape[1] // 2 + 1),:]
        x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
        x = torch.fft.irfft(x, n=res_shape1)
    else:
        x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
        x = torch.fft.irfft(x)
    return x


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

    # Vc = torch.fft.rfft(v, 1, onesided=False)
    Vc = _rfft(v, 1, onesided=False)

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

    # v = torch.fft.irfft(V, 1, onesided=False)
    v = _irfft(V, 1, onesided=False)

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


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_3d(dct_3d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)




if __name__ == '__main__':
    x = torch.Tensor(1000,4096)
    x.normal_(0,1)
    linear_dct = LinearDCT(4096, 'dct')
    error = torch.abs(dct(x) - linear_dct(x))
    assert error.max() < 1e-3, (error, error.max())
    linear_idct = LinearDCT(4096, 'idct')
    error = torch.abs(idct(x) - linear_idct(x))
    assert error.max() < 1e-3, (error, error.max())