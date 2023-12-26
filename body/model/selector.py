from omegaconf import OmegaConf
from .demucs import Demucs
from .tunet import TUNet

from ..utils.creator_factory import makeClassMaker

def selector(args):
    """
    :params: args: actually args.model
    """
    b = OmegaConf.to_container(args)
    print("model\t:",b)
    if args.name == "demucs" or args.name == "Demucs":
        return makeClassMaker(Demucs, **b['parameters'])
    elif args.name == "baseline" or args.name == "TUNet" or args.name == "tunet":
        return makeClassMaker(TUNet)