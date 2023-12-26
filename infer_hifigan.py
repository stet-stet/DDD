import os 
import logging
import hydra
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

def run(args):
    import torch
    from torch.utils.data import DataLoader

    from body import LoaderSelector
    from body import AugmentorSelector
    from body import ModelSelector
    from body import LossSelector
    from body import OptimizerSelector
    from body import SolverSelector

    from body.solver.enhance_audio import enhance
    # TODO: scheduler

    torch.manual_seed(1000000007) # 1e+8 + 7
    
    model = ModelSelector(args.model)() 

    # load from designated path into this model.

    load_from = args.load_from
    package = torch.load(load_from, 'cpu')
    model.load_state_dict(package['model']['state'])

    # now let's run inference
    out_dir = args.out_dir
    enhance(args, model, out_dir)

def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    run(args)

@hydra.main(version_base=None,config_path="conf")
def main(args):
    try:
        os.makedirs(f"{args.experiment.output_path}",exist_ok=True)
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        os._exit(1)

if __name__ == "__main__":
    main()






