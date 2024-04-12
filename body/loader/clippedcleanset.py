# Adapted from https://github.com/facebookresearch/denoiser/, under CC BY-NC 4.0 license
#    The corresponding LICENSE can be found on the incl_licenses directory.

import json
import logging
import os
import re

from .audio import Audioset
from .audio_randomhardclip import RandomHardclipAudioset

logger = logging.getLogger(__name__)


def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.

    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    else:
        raise ValueError(f"Invalid value for matching {matching}")


class ClippedCleanSet:
    def __init__(self, json_dir, matching="sort", segment=2.0, stride=0.5,
                 pad=True, sample_rate=None, valid_length_func=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param segment: maximum sequence length (s)
        :param stride: the stride used for splitting audio sequences (s)
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        :param valid_length_func: some models require a specific # of samples to avoid 0-padding. 
                                  this is a function to evaluate this
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        match_files(noisy, clean, matching)
        length = int(sample_rate * segment) if segment is not None else None
        stride = int(sample_rate * stride) if segment is not None else None
        if valid_length_func is not None and length is not None:
            length = valid_length_func(length)
        print("lenngth", length)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = RandomHardclipAudioset(clean, **kw) # Not a typo! this is going to be clipped

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)
