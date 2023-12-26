from .bandmask import BandMask
from .remix import Remix
from .revecho import RevEcho
from .shift import Shift

from omegaconf import OmegaConf
import torch

from ..utils.creator_factory import makeClassMaker

def selector(args):
    """
    :params: args: actually args.augmentor.
    """
    b = OmegaConf.to_container(args)
    # this should be a list of things...
    print("augmentor\t:", b)
    augments = []
    for profile in b:
        if profile['name'] == "remix":
            augments.append(Remix())
        elif profile['name'] == "bandmask":
            augments.append(BandMask(**profile['params']))
        elif profile['name'] == "shift":
            augments.append(Shift(**profile['params']))
        elif profile['name'] == "revecho":
            augments.append(RevEcho(**profile['params']))
    def makeWholeChain():
        return torch.nn.Sequential(*augments)
    return makeWholeChain



