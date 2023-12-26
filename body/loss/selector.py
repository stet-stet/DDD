from omegaconf import OmegaConf
from .stft_l1_loss import MultiResolutionSTFTL1Loss
from ..utils.creator_factory import makeClassMaker
import torch.nn as nn

def selector(args):
    """
    :params: args: actually args.augmentor
    """
    b = OmegaConf.to_container(args)
    print("loss\t:",b)
    if args.name == "MultiResolutionSTFTL1Loss":
        return makeClassMaker(MultiResolutionSTFTL1Loss, **b['parameters'])
    elif args.name == "L1Loss" or args.name == "L1loss":
        return makeClassMaker(nn.L1Loss, **b['parameters'])