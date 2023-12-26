import os 
import logging
import hydra
from omegaconf import OmegaConf
import itertools

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
    # TODO: scheduler

    torch.manual_seed(1000000007) # 1e+8 + 7
    
    augmentor = AugmentorSelector(args.augmentor)()
    generator = ModelSelector(args.model)().to('cuda')
    #discriminator1 = ModelSelector(args.model.discriminator1)().to('cuda')
    #discriminator2 = ModelSelector(args.model.discriminator2)().to('cuda') both inside

    tr_loader = DataLoader(
        LoaderSelector(args.tr_loader)(args.data.train,valid_length_func=generator.valid_length),
        batch_size=args.experiment.batch_size.tr, shuffle=True)
    cv_loader = DataLoader(
        LoaderSelector(args.cv_loader)(args.data.valid,valid_length_func=generator.valid_length),
        batch_size=args.experiment.batch_size.cv, shuffle=False)
    tt_loader = DataLoader(
        LoaderSelector(args.ts_loader)(args.data.test ,valid_length_func=generator.valid_length),
        batch_size=args.experiment.batch_size.ts, shuffle=False)

    generator_loss = LossSelector(args.loss)().to('cuda')

    generator_optimizer = OptimizerSelector(args.optimizer)(generator.parameters())
    #discriminator1_optimizer = OptimizerSelector(args.optimizer)(discriminator1.parameters())
    #discriminator2_optimizer = OptimizerSelector(args.optimizer)(discriminator2.parameters())

    solver = SolverSelector(args.solver)(
        tr_loader, cv_loader, tt_loader,
        generator, 
        generator_optimizer, 
        augmentor, 
        generator_loss,
        args # discriminators & optimizers are instantiated within the solver! the loss too.
    )
    solver.train()

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






