from omegaconf import OmegaConf

def selector(args):
    """
    :params: args: actually args.scheduler
    """
    print(OmegaConf.to_yaml(args))
    