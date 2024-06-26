# Adapted from https://github.com/facebookresearch/denoiser/, under CC BY-NC 4.0 license
#    The corresponding LICENSE can be found on the incl_licenses directory.
import random
import torch as th 

class Remix(th.nn.Module):
    """Remix.
    Mixes different noises with clean speech within a given batch
    """
    def forward(self, sources):
        noise, clean = sources
        bs, *other = noise.shape
        device = noise.device
        perm = th.argsort(th.rand(bs, device=device), dim=0)
        return th.stack([noise[perm], clean])
