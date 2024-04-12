# Adapted from an adaptation of Tomoki Hayashi's code, in https://github.com/facebookresearch/denoiser/
#   Original code by Tomoki Hayashi, MIT License (https://opensource.org/licenses/MIT)
#   Adaptation has CC BY-NC 4.0, found in the directory incl_licenses

"""
STFT-based Loss modules, perceptually weighted 
as suggested in https://arxiv.org/pdf/2101.07412.pdf
This didn't work that well and this did not make the final cut.
"""

import torch
import torch.nn.functional as F
import time

import numpy as np

alpha = [ 7.39626591e-01,  2.31218820e-02,  4.95042955e-02, -6.13830731e-03,
       -8.98332984e-03,  1.11515565e-02,  8.79213940e-03, -1.65436826e-02,
       -9.47038093e-03, -2.35939689e-02, -2.93726877e-02, -1.46480216e-02,
       -5.02206351e-04, -9.54786512e-03, -1.41396074e-03,  1.11708468e-02,
        7.98036588e-03,  9.89428100e-04, -7.40858139e-03,  2.93585690e-04,
        6.35048368e-03,  4.20239164e-03, -6.91803132e-04, -6.92533698e-04,
        1.77558679e-04, -6.47081869e-04, -4.04865593e-04,  3.25472147e-04,
        5.38711640e-04, -3.23794134e-03, -6.08985213e-03, -1.72301599e-02]

def make_fourier_mags(pole_coeffs, ticks): # use n_fft//2+1 ticks
    frequencies = np.linspace(0, np.pi, ticks)
    multiply_every_time = np.exp( -1j * frequencies)
    current = np.ones((ticks,),dtype=np.complex128)
    ret = np.ones(ticks,dtype=np.complex128)
    for coeff in pole_coeffs:
        ret -= current * coeff
        current *= multiply_every_time
    return np.abs(ret)

# alpha-1 to alpha-32
# a 32-degree all-pole filter whose parameters were derived from clean_trainset_wav.
# procedure: divide voice into segments of length 1024, fft them, average their magnitude, irfft, then do:
## degree = 32
## mat = np.linalg.inv([[get_autocorr(yyy,np.abs(i-j)) for i in range(degree)] for j in range(degree)])
## vec = np.array([get_autocorr(yyy,i) for i in range(1,1+degree)])
## alpha = np.matmul(mat,vec) 

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag, weights):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            weights (Tensor): Perceptual weight of dimension (#freq_bins, )
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm((y_mag - x_mag)*weights, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag, weights):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
            weights (Tensor): Perceptual weight of dimension (#freq_bins, ).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag)*torch.log(weights), torch.log(x_mag)*torch.log(weights)) 


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.weights = make_fourier_mags(alpha, fft_size//2 + 1)
        self.weights = torch.Tensor(self.weights / self.weights.max() ).to('cuda:0')
        self.weights.requires_grad_(False)

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag, self.weights)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag, self.weights)

        return sc_loss, mag_loss


class PerceptuallyWeightedMultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(PerceptuallyWeightedMultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        print("using perceptually weighted loss:")
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss

