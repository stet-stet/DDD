from omegaconf import OmegaConf
from .noisycleanset import NoisyCleanSet
from .clippedcleanset import ClippedCleanSet

from ..utils.creator_factory import makeClassMaker

def selector(args):
    """
    :params: args: actually either args.tr_loader, args.cv_loader args.ts_loaser
    """
    print("dataset\t:",args)
    b = OmegaConf.to_container(args)
    print("loader\t:", b)
    if args.name == "NoisyCleanSet":
        return makeClassMaker(NoisyCleanSet, **b['parameters'])
    if args.name == "ClippedCleanSet":
        return makeClassMaker(ClippedCleanSet, **b['parameters'])