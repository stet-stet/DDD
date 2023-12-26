from omegaconf import OmegaConf
import torch 
from ..utils.creator_factory import makeClassMaker

def selector(args):
    """
    :params: args: actually args.optimizer
    """
    b = OmegaConf.to_container(args)
    if args.name == "adam" or args.name=="Adam":
        return makeClassMaker(torch.optim.Adam, 
                              lr=b['parameters']['lr'], 
                              betas=(b['parameters']['beta1'],b['parameters']['beta2']) )
    if args.name == "AdamW" or args.name=="adamw":
        return makeClassMaker(torch.optim.AdamW,
                              lr=b['parameters']['lr'],
                              betas=(b['parameters']['beta1'],b['parameters']['beta2']), 
                              weight_decay=b['parameters']['weight_decay'])