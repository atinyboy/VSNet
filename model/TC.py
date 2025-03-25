import torch
import torch.nn as nn


class TemporalConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='causal', num_nodes=25):
        super(TemporalConvLayer, self).__init__()
        self.num_nodes = num_nodes
        if padding == 'causal':
            self.padding = (kernel_size - 1)
        else:
            self.padding = 0

        # Use Conv2d where the convolution is applied along the time dimension (T)
        self.conv = nn.Conv2d(
            in_channels=in_channels,  # Channels (C)
            out_channels=out_channels,  # Output channels
            kernel_size=(kernel_size, 1),  # Convolve only along T, not V
            stride=(stride, 1),  # Stride only along T
            padding=(0, 0)  # No padding here; handled manually if causal
        )

    def forward(self, x):

        # Apply causal padding if necessary
        if self.padding > 0:
            x = nn.functional.pad(x, (0, 0, self.padding, 0))  # Pad only on T dimension

        # Perform convolution
        out = self.conv(x)

        return out



