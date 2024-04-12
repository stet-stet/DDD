# Adapted from https://github.com/facebookresearch/denoiser/, under CC BY-NC 4.0 license
#    The corresponding LICENSE can be found on the incl_licenses directory.

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

from ..model.hifigan_discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator

from .. import OptimizerSelector

logger = logging.getLogger(__name__)

def benchmark_benchmark(benchmark, tp, string):
    if benchmark:
        logger.info(f'{time.time() - tp} between {string}')
    return time.time()

# The below are losses unique to these discriminators
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr,dg):
            loss += torch.mean(torch.abs(rl-gl))
    return loss

def generator_loss(disc_outputs):
    loss = 0
    #gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        #gen_losses.append(l)
        loss += l
    #return loss, gen_losses # <- is this actually needed?
    return loss

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss+g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses

class HiFiGANAudioToAudioSolver(object):
    def __init__(self, 
                tr_loader, cv_loader, tt_loader, 
                generator,
                generator_optimizer,
                augmentor, 
                user_loss,
                args):
        self.device = "cuda" if torch.cuda.is_available else "cpu"

        self.tr_loader = tr_loader 
        self.cv_loader = cv_loader 
        self.tt_loader = tt_loader

        self.model = {
            "generator": generator,
            "msd": MultiScaleDiscriminator().to(self.device),
            "mpd": MultiPeriodDiscriminator().to(self.device)
        }

        self.optimizer = {
            "generator": generator_optimizer,
            "msd": OptimizerSelector(args.optimizer)(self.model['msd'].parameters()),
            "mpd": OptimizerSelector(args.optimizer)(self.model['mpd'].parameters())
        }

        self.augment = augmentor

        self.user_loss = user_loss

        self.epochs = args.experiment.epochs

        # TODO: checkpoints are a 3-tuple of paths (since there are three models)
        self.continue_from = '' # XXX
        self.eval_every = args.experiment.eval_every
        self.best_file = Path(f"{args.experiment.output_path}/best.th")
        self.checkpoint_file = Path(f"{args.experiment.output_path}/checkpoint.th") 
        self.save_every_file = Path(f"{args.experiment.output_path}/epoch.th")
        self.history_file = Path(f"{args.experiment.output_path}/{args.experiment.history_file}")

        self.best_state = {key:None for key in self.model}
        self.current_state = {key:None for key in self.model}
        self.history = []
        self.samples_dir = f"{args.experiment.output_path}/samples" # where to save samples
        self.num_prints = args.experiment.num_prints
        self.save_every = args.experiment.save_every
        self.args = args
        
        self._reset()

    def _serialize(self, epoch=None):
        package = {"model":{}, "optimizer":{}, "best_state":{}}
        for key in self.model:
            package['model'][key] = serialize_model(self.model[key])
            package['optimizer'][key] = self.optimizer[key].state_dict()
            package['best_state'][key] = self.best_state[key]
        package['history'] = self.history    
        package['args'] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        model = package['model']
        model['state'] = {}
        for key in self.model:
            model['state'][key] = self.best_state[key]
        tmp_path = str(self.best_file) + ".tmp"
        torch.save(model, tmp_path)
        os.rename(tmp_path, self.best_file)

        if epoch % self.save_every == 0:
            model = package['model']
            for key in self.model:
                model['state'][key] = self.current_state[key]
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
            package = torch.load(load_from,'cpu')
            if load_best:
                for key in self.model:
                    self.model[key].load_state_dict(package['best_state'][key])
            else:
                for key in self.model:
                    self.model[key].load_state_dict(package['model'][key]['state'])
            if 'optimizer' in package and not load_best:
                for key in self.model:
                    self.optimizer[key].load_state_dict(package['optimizer'][key])
            if keep_history:
                self.history = package['history']
            for key in self.model:
                self.best_state[key] = package['best_state'][key]
        continue_pretrained = self.args.experiment.continue_pretrained
        if continue_pretrained:
            logger.info("Fine tuning from pre-trained model %s", continue_pretrained)
            model = getattr(pretrained, self.args.experiment.continue_pretrained)()
            for key in self.model:
                self.model[key].load_state_dict(model[key].state_dict())
    
    def train(self):
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")
        
        # TODO: scheduler should also be progressed accordingly
        logger.info("Making sure that cross-validation, testing, enhancing does not break...")

        # pesq, stoi = evaluate(self.args, self.model["generator"], self.tt_loader)
        # enhance(self.args, self.model['generator'], self.samples_dir)
        # with torch.no_grad():
        #     valid_loss = self._run_one_epoch(101, cross_valid=True)
        # self._serialize(0)

        for epoch in range(len(self.history), self.epochs):
            for key in self.model:
                self.model[key].train()
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
                for key in self.model:
                    self.model[key].eval()
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
                for key in self.model:
                    self.best_state[key] = copy_state(self.model[key].state_dict())
            for key in self.model:
                self.current_state[key] = copy_state(self.model[key].state_dict())

            # evaluate samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:
                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # # We used to switch to the best known model for testing:
                # with swap_state(self.model, self.best_state):
                #    pesq, stoi = evaluate(self.args, self.model, self.tt_loader)

                pesq, stoi = evaluate(self.args, self.model["generator"], self.tt_loader)

                metrics.update({'pesq': pesq, 'stoi': stoi})
                logger.info('Enhance and save samples...')
                enhance(self.args, self.model['generator'], self.samples_dir)
                        

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
            tp = benchmark_benchmark(benchmark, tp, "to load data")
            if not cross_valid:
                sources = torch.stack([noisy - clean, clean])
                sources = self.augment(sources)
                noise, clean = sources
                noisy = noise + clean
            tp = benchmark_benchmark(benchmark, tp, "for augmentation")
            
            y_hat = self.model["generator"](noisy)

            tp = benchmark_benchmark(benchmark, tp, "for forward pass")
            if not cross_valid:
                self.optimizer['mpd'].zero_grad()
                self.optimizer['msd'].zero_grad()
                #MPD
                y_df_hat_r,y_df_hat_g,_,_ = self.model['mpd'](clean, y_hat.detach())
                loss_disc_f, _, _ = discriminator_loss(y_df_hat_r, y_df_hat_g)
                #MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = self.model['msd'](clean, y_hat.detach())
                loss_disc_s, _, _ = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                (loss_disc_s + loss_disc_f).backward()
                self.optimizer['mpd'].step()
                self.optimizer['msd'].step()

                self.optimizer['generator'].zero_grad()

                loss_gen = self.user_loss(y_hat, clean) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.model['mpd'](clean, y_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.model['msd'](clean, y_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g) * 2
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g) * 2
                loss_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s = generator_loss(y_ds_hat_g)
                (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_gen).backward()

                self.optimizer['generator'].step()
            else: # is cross_valid
                loss_gen = self.user_loss(y_hat, clean) * 45

            tp = benchmark_benchmark(benchmark, tp, "for backward pass")

            total_loss += loss_gen.item()
            logprog.update(loss=format(total_loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del loss_gen
        return total_loss / (i + 1)
