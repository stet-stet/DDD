# Adapted from https://github.com/facebookresearch/denoiser/, under CC BY-NC 4.0 license
#    The corresponding LICENSE can be found on the incl_licenses directory.
import random
import torch as th
from torch.nn import functional as F

from ..utils import dsp


class BandMask(th.nn.Module):
    """BandMask.
    Maskes bands of frequencies. Similar to Park, Daniel S., et al.
    "Specaugment: A simple data augmentation method for automatic speech recognition."
    (https://arxiv.org/pdf/1904.08779.pdf) but over the waveform.
    """

    def __init__(self, max_width=0.2, bands=120, sample_rate=16_000):
        """__init__.
        :param max_width: the maximum width to remove
        :param bands: number of bands
        :param sample_rate: signal sample rate
        """
        super().__init__()
        self.max_width = max_width
        self.bands = bands
        self.sample_rate = sample_rate

    def forward(self, wav):
        bands = self.bands
        bandwidth = int(abs(self.max_width) * bands)
        mels = dsp.mel_frequencies(bands, 40, self.sample_rate/2) / self.sample_rate
        low = random.randrange(bands)
        high = random.randrange(low, min(bands, low + bandwidth))
        filters = dsp.LowPassFilters([mels[low], mels[high]]).to(wav.device)
        low, midlow = filters(wav)
        # band pass filtering
        out = wav - midlow + low
        return out