import torch
import torch.nn.functional as F 
import time 
from .stft_loss import MultiResolutionSTFTLoss
from .stft_perceptual_weighted_loss import PerceptuallyWeightedMultiResolutionSTFTLoss

class MultiResolutionSTFTL1Loss(torch.nn.Module):
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1, 
                 ratio=1,
                 perceptual_weighting=False):
        super(MultiResolutionSTFTL1Loss,self).__init__()
        if perceptual_weighting is False:
            self.mrstftloss = MultiResolutionSTFTLoss(fft_sizes=fft_sizes,
                                                  hop_sizes=hop_sizes,
                                                  win_lengths=win_lengths,
                                                  window=window, factor_sc=factor_sc, factor_mag=factor_mag).to('cuda:0')
        else:
            self.mrstftloss = PerceptuallyWeightedMultiResolutionSTFTLoss(fft_sizes=fft_sizes,
                                                  hop_sizes=hop_sizes,
                                                  win_lengths=win_lengths,
                                                  window=window, factor_sc=factor_sc, factor_mag=factor_mag).to('cuda:0')
        self.l1loss = torch.nn.L1Loss(reduction='mean').to('cuda:0')
        self.ratio = ratio

    def forward(self, x, y):
        """
        x: prediction, (B, 1, T)
        y: GT, (B, 1, T)
        """
        tp = time.time()
        sc_loss, mag_loss = self.mrstftloss(x.squeeze(1), y.squeeze(1))
        tp = time.time()
        loss = self.l1loss(x,y)
        tp = time.time()
        return loss + self.ratio * (sc_loss + mag_loss)

