# DDD

Due to the original repo containing some sensitive private information, we decided to disclose our code as a separate public repo.

## Setting up

Make a python 3.8 venv, and then:

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
```

## Reproducing Results

Please download the checkpoints, and take note of where the checkpoints are.

Moreover, please download the pre-clipped test-sets as well. (This way, we won't have headaches about file structure.)

For DDD and 1dB(voicebank-demand):
```
python infer_gan.py \
  +noisy_dir=(directory with dataset)/vbdm-clipped/clipped_1dB \
  +noisy_json= \
  +load_from=(directory with models)/hifigan_bs2/checkpoint.th \
  +out_dir=inferred/1dB/hifigan_bs2 \
  +experiment=setting_3 \
  +model=setting_1
```

For DD:
```
python infer_vanilla.py \
  +noisy_dir=(directory with dataset)/vbdm-clipped/clipped_1dB \
  +noisy_json= \
  +load_from=(directory with models)/shift_only_adamw/checkpoint.th \
  +out_dir=inferred/1dB/shift_only_adamw \
  +experiment=setting_3 \
  +model=setting_1
```

For T-Unet:
```
python infer_vanilla.py \
  +noisy_dir=(directory with dataset)/vbdm-clipped/16384_clipped_1dB \
  +noisy_json= \
  +load_from=(directory with models)/realbaseline/checkpoint.th \
  +out_dir=inferred/1dB/realbaseline \
  +experiment=setting_3 \
  +model=realbaseline
```

The "real" in `realbaseline` comes from how we initially made a mistake in copying the T-UNet architecture and then had to modify code after a correspondence with Dr. Nair (the first author of their paper).

Modify the scripts as necessary. Alternatively, please look at the `runinfer_in_batch*` files and use them accordingly.

## Training the Models

Step 1: Download the Voicebank-DEMAND dataset and decompress it. Delete the noisy split. Resample all to 16kHz. In a separate directory, clip the clean speech from the test set so that they have a SNR of 3dB. 

Alternatively, I have prepared a reproduction kit where all of this is already done for you. Let the path to these samples be `DATA`.



Step 2: Run the following
```
noisy_train=$DATA/clean_trainset_wav
clean_train=$DATA/clean_trainset_wav
noisy_test=$DATA/clipped_3dB
clean_test=$DATA/clean_testset_wav
noisy_dev=$DATA/clean_testset_wav
clean_dev=$DATA/clean_testset_wav

mkdir -p egs/vbdm_clip/tr
mkdir -p egs/vbdm_clip/cv
mkdir -p egs/vbdm_clip/ts

python make_egs.py $noisy_train > egs/vbdm_clip/tr/noisy.json
python make_egs.py $clean_train > egs/vbdm_clip/tr/clean.json
python make_egs.py $noisy_test > egs/vbdm_clip/ts/noisy.json
python make_egs.py $clean_test > egs/vbdm_clip/ts/clean.json
python make_egs.py $noisy_dev > egs/vbdm_clip/cv/noisy.json
python make_egs.py $clean_dev > egs/vbdm_clip/cv/clean.json

```

Confirm that the resulting json files are nonempty.

Step 3: run the script.

For DDD: `runmain_clipped_hifigan.sh`.

For DD: `runmain.sh`.

For T-UNet: `runrealbaseline.sh`.

## Acknowledgement

The structure of the code, was heavily inspired by [this](https://github.com/facebookresearch/denoiser) repository, where Demucs is used to perform speech denoising.

## Goals and Features

- each training component is fully modular
  - ...so that minimal number of files need be modified to run a different experiment
- a component can be added without much hassle (as long as input/output dimensions match)
- running an experiment can be done without much hassle
- easily keep track of which experiments resulted in which file (the logger does this for you!)
- a code that can be easily inspected even for beginners!

## The main idea

- Hydra shall be used to implement the below.
- augmentors, loaders, losses, models, optimizers, lr-schedulers are chosen when running file
- each config outlines:
  - parameters for each component
  - (when not specified, parameter defaults to defaults)
- there is one "selector" function within each folder: this is the only func that needs to be imported
  - the json parameters will be directly fed into the component constructor!
- the training function will have to be hand-written for each different experiment
  - in `experiments/`
- output folder, logs, etc are automatically generated such that no information goes deleted without a trace


