# DDD

Due to the original repo containing some sensitive private information, we decided to disclose our code as a separate public repo.

## Setting up

Make a python 3.8 venv, and then:

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir
```

## Generated Samples

Genereated Samples are available [here](https://drive.google.com/file/d/1b3Id8LVFHVhs5SpxI9emvUOt3ofbDHIz/view?usp=sharing). As outlined on the paper, these samples were all normalized with pyloudnorm to -27 LUFS before they could be used in subjective testing.

Alternatively, you can listen to some samples [on our demo page](https://stet-stet.github.io/DDD/). The audio on this website has been normalized to -27 LUFS. (All normalization was done with [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm). Awesome package!)

## Reproducing Results

Please download the checkpoints [link](https://drive.google.com/file/d/1UeJLcp3riu5MiB0mgQ-vS4zI3yCoqY62/view?usp=sharing) and the "reproduction kit" which contains parts of the dataset we used [link](https://drive.google.com/file/d/1-cXl2RSreqYQLv-yNLaDnsa3eKEiC1hJ/view?usp=sharing). We used only one split of each dataset and down-sampled them, so it is strongly recommended that you download them. After downloading, place the reproduction kit in `data`.

For DDD and 1dB(voicebank-demand):
```
python infer_gan.py \
  +noisy_dir=data/vbdm/clipped_1dB \
  +noisy_json= \
  +load_from=(directory with models)/hifigan_bs2/checkpoint.th \
  +out_dir=inferred/1dB/hifigan_bs2 \
  +experiment=setting_3 \
  +model=setting_1
```

For DD:
```
python infer_vanilla.py \
  +noisy_dir=data/vbdm/clipped_1dB \
  +noisy_json= \
  +load_from=(directory with models)/shift_only_adamw/checkpoint.th \
  +out_dir=inferred/1dB/shift_only_adamw \
  +experiment=setting_3 \
  +model=setting_1
```

For T-Unet:
```
python infer_vanilla.py \
  +noisy_dir=data/vbdm/16384_clipped_1dB \
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

Alternatively, I have prepared a reproduction kit where all of this is already done for you.

Step 2: 
```
noisy_train=data/clean_trainset_wav
clean_train=data/clean_trainset_wav
noisy_test=data/clipped_3dB
clean_test=data/clean_testset_wav
noisy_dev=data/clean_testset_wav
clean_dev=data/clean_testset_wav

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

Confirm that the resulting json files are nonempty, and each entry contains the absolute path of the speech samples.

Step 3: run the script.

For DDD: `runmain_clipped_hifigan.sh`.

For DD: `runmain.sh`.

For T-UNet: `runrealbaseline.sh`.

## Acknowledgement & Clarification on License

The structure of the code, was heavily inspired by [this](https://github.com/facebookresearch/denoiser) repository, where Demucs is used to perform speech denoising. That repo is licensed under CC BY-NC 4.0, included in this repo under the directory `incl_licenses`. While other codes are licensed under MIT, please understand that a large portion of this repo is still under CC BY-NC 4.0 as per the restrictions on the repo I drew from.

