import json
import logging
from pathlib import Path
import os
import time
import datetime

import torch
import torch.nn.functional as F

# TODO: write SchedulerSelector & incorporate scheduler into code

from .evaluate_audio import evaluate
from .enhance_audio import enhance
from ..utils.training import serialize_model, swap_state, copy_state, pull_metric
from ..utils.logging import bold, LogProgress

logger = logging.getLogger(__name__)

def benchmark_benchmark(benchmark, tp, string):
    if benchmark:
        logger.info(f'{time.time() - tp} between {string}')
    return time.time()

class AudioToAudioSolver(object):
    def __init__(self, tr_loader, cv_loader, tt_loader, model, optimizer, augmentor, loss, args):
        self.device = "cuda" if torch.cuda.is_available else "cpu"

        self.tr_loader = tr_loader 
        self.cv_loader = cv_loader 
        self.tt_loader = tt_loader
        self.model = model
        self.optimizer = optimizer
        self.augment = augmentor
        self.loss = loss

        self.epochs = args.experiment.epochs

        # TODO: checkpoints
        self.continue_from = '' # XXX
        self.eval_every = args.experiment.eval_every
        self.best_file = Path(f"{args.experiment.output_path}/best.th")
        self.checkpoint_file = Path(f"{args.experiment.output_path}/checkpoint.th")
        self.save_every_file = Path(f"{args.experiment.output_path}/epoch.th")
        self.history_file = Path(f"{args.experiment.output_path}/{args.experiment.history_file}")

        self.best_state = None
        self.current_state = None
        self.history = []
        self.samples_dir = "samples" # where to save samples
        self.num_prints = args.experiment.num_prints
        self.save_every = args.experiment.save_every
        self.args = args
        
        self._reset()

    def _serialize(self, epoch=None):
        package = {}
        package['model'] = serialize_model(self.model)
        package['optimizer'] = self.optimizer.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        model = package['model']
        model['state'] = self.best_state
        tmp_path = str(self.best_file) + ".tmp"
        torch.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

        if epoch % self.save_every == 0:
            model = package['model']
            model['state'] = self.current_state
            if epoch is None:
                tmp_path = datetime.datetime.now().strftime("%y-%m-%H-%M-%S") + ".tmp"
            else:
                tmp_path = str(epoch) + ".tmp"
            torch.save(model, tmp_path)
            os.rename(tmp_path, f"{self.save_every_file}_{epoch}.th")
        
    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint_file.exists():
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = False
            keep_history = False

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            if load_best:
                self.model.load_state_dict(package['best_state'])
            else:
                self.model.load_state_dict(package['model']['state'])
            if 'optimizer' in package and not load_best:
                self.optimizer.load_state_dict(package['optimizer'])
            if keep_history:
                self.history = package['history']
            self.best_state = package['best_state']
        continue_pretrained = self.args.experiment.continue_pretrained
        if continue_pretrained:
            logger.info("Fine tuning from pre-trained model %s", continue_pretrained)
            model = getattr(pretrained, self.args.experiment.continue_pretrained)()
            self.model.load_state_dict(model.state_dict())
    
    def train(self):
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")
        
        # TODO: scheduler should also be progressed accordingly

        logger.info("Making sure that cross-validation, testing, enhancing does not break...")
        self._serialize(0)


        for epoch in range(len(self.history), self.epochs):
            self.model.train()
            start = time.time()
            logger.info('-'*70)
            logger.info("training...")
            train_loss = self._run_one_epoch(epoch)
            logger.info(
                bold(f'Train Summary | End of Epoch {epoch + 1} | '
                     f'Time {time.time() - start:.2f}s | Train Loss {train_loss:.5f}'))
            
            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    valid_loss = self._run_one_epoch(epoch, cross_valid=True)
                logger.info(
                    bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                         f'Time {time.time() - start:.2f}s | Valid Loss {valid_loss:.5f}'))
            else:
                valid_loss = 0

            best_loss = min(pull_metric(self.history, 'valid') + [valid_loss])
            metrics = {'train': train_loss, 'valid': valid_loss, 'best': best_loss}
            if valid_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), valid_loss)
                self.best_state = copy_state(self.model.state_dict())
            self.current_state = copy_state(self.model.state_dict())

            # evaluate samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # # We used to switch to the best known model for testing:
                # with swap_state(self.model, self.best_state):
                #    pesq, stoi = evaluate(self.args, self.model, self.tt_loader)

                pesq, stoi = evaluate(self.args, self.model, self.tt_loader)

                metrics.update({'pesq': pesq, 'stoi': stoi})

                # TODO enhance and save samples                

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            json.dump(self.history, open(self.history_file, "w"), indent=2)
            self._serialize(epoch + 1)
            logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())

    def _run_one_epoch(self, epoch, cross_valid=False):
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            benchmark = (i%2000 == 30)
            tp = time.time()
            noisy, clean = [x.to(self.device) for x in data]
            tp = benchmark_benchmark(benchmark, tp, "init& data loading")
            if not cross_valid:
                sources = torch.stack([noisy - clean, clean])
                sources = self.augment(sources)
                noise, clean = sources
                noisy = noise + clean
            tp = benchmark_benchmark(benchmark, tp, "data loading & aug")
            estimate = self.model(noisy)
            tp = benchmark_benchmark(benchmark, tp, "aug & forward pass")
            # apply a loss function after each layer
            with torch.autograd.set_detect_anomaly(True):
                # optimize model in training mode
                the_loss = self.loss(estimate, clean)
                tp = benchmark_benchmark(benchmark, tp, "forward pass & loss")
                if not cross_valid:
                    self.optimizer.zero_grad()
                    the_loss.backward()
                    self.optimizer.step()
                tp = benchmark_benchmark(benchmark, tp, "loss & backprop")

            total_loss += the_loss.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del the_loss, estimate
        return total_loss / (i + 1)
