from omegaconf import OmegaConf
from .audio_to_audio import AudioToAudioSolver
from .hifigan_audio_to_audio import HiFiGANAudioToAudioSolver
from ..utils.creator_factory import makeClassMaker

def selector(args):
    """
    :params: args: actually args.solver
    """
    b = OmegaConf.to_container(args)
    print("solver\t:",b)
    if args.name == "AudioToAudioSolver":
        return makeClassMaker(AudioToAudioSolver)
    elif args.name == "HiFiGANAudioToAudioSolver":
        return makeClassMaker(HiFiGANAudioToAudioSolver)
    raise Exception("The Solver selected does not exist!")