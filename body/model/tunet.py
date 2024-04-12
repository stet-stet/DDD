import math
import time

import torch
from torch import nn
from torch.nn import functional as F

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output

    taken from https://github.com/serkansulun/pytorch-pixelshuffle1d/blob/master/pixelshuffle1d.py
    """
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor
    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]
        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width
        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        return x

class TUNet(nn.Module):
    """
    Baseline model for audio declipping, as outlined and used in 
    "Cascaded time+ time-frequency unet for speech enhancement: Jointly addressing clipping, codec distortions, and gaps (ICASSP 2021)".
    The model architecture was determined to be the below after an e-mail correspondence with Dr. Nair, the first author of above.

    the model is a slightly twisted SEGAN.
    the model operates in 16000Hz, and only takes signals of length 16384 as input.

    Args: None. configuration is fixed.
    """
    def __init__(self):
        super().__init__()
        self.sample_rate = 16000
        self.chin = 1

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        encoder_channels_after_each_layer = [
            1, 64, 128,128,128,128,128,128, 256,256,256,256,256,256, 512,
        ]
        decoder_channels_after_each_layer = [
            512, 256,256,256,256,256,256, 128,128,128,128,128,128, 64, 1
        ]

        for index in range(13):
            encode = []
            encode += [
                nn.Conv1d(encoder_channels_after_each_layer[index],
                        encoder_channels_after_each_layer[index+1],
                        5, #kernel size
                        stride=(2 if index!=0 else 1), padding=2), #stride
                nn.BatchNorm1d(encoder_channels_after_each_layer[index+1]),
                nn.LeakyReLU(negative_slope=0.1),
            ]
            self.encoder.append(nn.Sequential(*encode))

        for index in range(13,14):
            # XXX: fig 1 of original paper makes it seem like the last encoder has a BN
            # but this should not be possible, as the encoder dimensions are (b, 512, **1**).
            encode = []
            encode += [
                nn.Conv1d(encoder_channels_after_each_layer[index],
                          encoder_channels_after_each_layer[index+1],
                          4, #kernel size
                          stride=1, padding=0),
                nn.LeakyReLU(negative_slope=0.1),
            ]
            self.encoder.append(nn.Sequential(*encode))

        for index in range(0,1):
            decode = []
            decode += [
                nn.Conv1d(decoder_channels_after_each_layer[index],
                        decoder_channels_after_each_layer[index+1]*4, # this one has to have a stride of 1/4
                        1, #kernel size
                        stride=1, padding=0), #stride, padding
                PixelShuffle1D(4),
                nn.BatchNorm1d(decoder_channels_after_each_layer[index+1]),
                nn.LeakyReLU(negative_slope=0.1),
            ]
            self.decoder.append(nn.Sequential(*decode))

        for index in range(1,13):
            decode = []
            decode += [
                nn.Conv1d(decoder_channels_after_each_layer[index],
                        decoder_channels_after_each_layer[index+1]*2,
                        5, #kernel size
                        stride=1, padding=2), #stride, padding
                PixelShuffle1D(2),
                nn.BatchNorm1d(decoder_channels_after_each_layer[index+1]),
                nn.LeakyReLU(negative_slope=0.1),
            ]
            self.decoder.append(nn.Sequential(*decode))

        for index in range(13,14): 
            # NOTE: fig 1 of original paper makes it seem like the last layer is not a sub-pixel convolution
            # however, if this were true, the dimensions would not align.
            decode = []
            decode += [nn.Conv1d(decoder_channels_after_each_layer[index],
                                 decoder_channels_after_each_layer[index+1],
                                 5, #kernel size
                                 stride=1, padding=2)]
            self.decoder.append(nn.Sequential(*decode))

    def valid_length(self, length):
        """
        Return the nearest valid length to use in model training.
        i.e. make the length (a multiple of 16384) + ("shift" (augmentation))

        if the mixture has a valid length, the estimated sources 
        will have exactly the same length.
        """
        return 16384 * (length // 16384) + 8000

    def valid_length_t(self, length):
        return 16384 * (length // 16384)
    
    def forward(self, wave):
        """
        params:
        - wave (torch.Tensor of shape (b?, 1, 16384*n))
        output:
        torch.Tensor of shape (b, 1, 16384*n)
        """
        assert wave.shape[-1] % 16384 == 0

        if wave.dim() == 2:
            wave = wave.unsqueeze(1)
        # TODO: reshape this into (b*n,1,16384), run forward, reshape and return
        b = wave.shape[0]
        wave = torch.reshape(wave, (wave.shape[0],1,-1,16384))
        n = wave.shape[2]
        wave = wave.permute(0,2,1,3)
        wave = torch.reshape(wave, (b*n,1, 16384))
        out = self.forward_16384(wave)
        out = torch.reshape(out, (b,n,1,16384))
        out = out.permute(0,2,1,3)
        out = torch.reshape(out, (b,1,n*16384))
        return out

    def forward_16384(self, wave):
        """
        params:
        - wave (torch.Tensor of shape (b, 1, 16384))
        output: torch.Tensor of shape (b, 1, 16384)
        """
        assert wave.shape[-1] == 16384
        skips = []
        x = wave
        for n,encode in enumerate(self.encoder):
            x = encode(x)
            if n != 13:
                skips.append(x)
        
        for n,decode in enumerate(self.decoder):
            x = decode(x)
            if n != 13:
                skip = skips.pop(-1)
                x = x + skip[..., :x.shape[-1]]
        x = x[..., :16384]
        return x

class TUNetStreamer:
    def __init__(self, tunet_model, num_frames=1, latency_interval=500):
        """
        params

        :param tunet_model (TUNet): the model we feed the samples to.
        :param num_frames (int): the number of frames that will be fed into the model at once
        """
        self.tunet_model = tunet_model
        self.buffer = 3
        self.frame_length = 16384 #fixed
        self.process_length = num_frames * self.frame_length
        self.num_frames = num_frames
        
        self.frames = 0
        self.total_time = 0
        self.pending = torch.zeros(1, 0, device='cpu') # evaluation gon be in the CPU anyways

        self.how_many_samples_input = 0
        self.how_many_samples_output = 0
        self.input_counter = {k:-1 for k in range(0,1638401,latency_interval)}
        self.input_pointer = 0
        self.output_counter = {k:-1 for k in range(0,1638401,latency_interval)}
        self.output_pointer = 0
        self.latency_interval = latency_interval

    def reset_time_per_frame(self):
        self.frames = 0
        self.total_time = 0

    @property
    def time_per_frame(self):
        return self.total_time / self.frames

    @property
    def average_latency(self):
        count, lat=0, 0
        for k in range(0,1638401,self.latency_interval):
            if self.output_counter[k]==-1: break
            count += 1
            lat += (self.output_counter[k] - self.input_counter[k])
        return lat / count

    def flush(self):
        """
        output zeros equivalent to self.pending.shape 
        also reset internal states.
        """
        self.reset_time_per_frame()
        return torch.zeros_like(self.pending,device=self.pending.device)

    def feed(self, wav):
        """
        feed wav to the streamer.
        if total length of self.pending goes over self.process_length, then this will return output equivalent to the length of processed samples.
        """
        assert len(wav.shape) == 2 # no putting in batches, no putting in raw array of samples; this has to be (1, len)

        start = time.time()
        self.pending = torch.cat([self.pending, wav], dim=1)
        self.how_many_samples_input += wav.shape[1]
        while self.input_pointer <= self.how_many_samples_input:
            self.input_counter[self.input_pointer] = start
            self.input_pointer += self.latency_interval

        out = torch.zeros(1,0,device='cpu')
        # FEEDING
        while self.pending.shape[1] >= self.process_length:
            self.frames += self.num_frames
            network_input = self.pending[:, :self.process_length]
            network_output = self.tunet_model(network_input).squeeze(0)
            self.pending = self.pending[:, self.process_length:]
            out = torch.cat([out, network_output], dim=1)
            self.how_many_samples_output += out.shape[1]

        end = time.time()
        while self.output_pointer <= self.how_many_samples_output:
            self.output_counter[self.output_pointer] = end
            self.output_pointer += self.latency_interval

        self.total_time += (end - start)
        
        return out


def test():
    net = TUNet()
    iii = torch.ones(1,1,16384)
    print(net(iii).shape)

    jjj= torch.ones(32, 1, 4*16384)
    print(net(jjj).shape)    
    
def stream_test():

    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(
        "body.model.tunet",
        description="Benchmark the T-UNet implementation. Conformance with offline implementation is not examined, as it is invariably zero."
                    "\n(note that such delta may arise only when upsampling is involved.)"
    )
    parser.add_argument("-f", "--num_frames", type=int, default=1)
    parser.add_argument("-i", "--latency_interval", type=int, default=500)
    parser.add_argument("-l","--benchmark_length", type=int, default=100)
    args = parser.parse_args()

    num_frames = args.num_frames
    latency_interval = args.latency_interval
    benchmark_length = args.benchmark_length
    net = TUNet()
    print("stream test begins")
    
    input_length = 16000 * benchmark_length
    time_per_step = latency_interval / 16000
    recorded_samples = torch.rand(1, input_length,device='cpu',pin_memory=True)
    streamer = TUNetStreamer(net,num_frames=num_frames,latency_interval=latency_interval)
    
    start_time=time.time()
    tp=0
    while tp < input_length:
        until=int((time.time()-start_time)*16000)
        streamer.feed(recorded_samples[:,tp:until])
        tp=until

    print(f"total time elasped in model: {streamer.total_time}")
    print(f"mean time-per-frame: {streamer.time_per_frame: 2f} (the lower the faster)")
    print(f"mean latency: {streamer.average_latency:2f} s")

def mac_test():
    from thop import profile
    model = TUNet().to('cpu')
    siglen = 131072
    input = torch.randn(1, siglen)
    macs, params = profile(model, inputs=(input, ))
    print("macs per sample", macs / siglen, params)


if __name__=="__main__":
    #test()
    #mac_test()
    stream_test()



            