import torch
import torch.nn as nn

class HumanSignalEncoder(nn.Module):
    """
    Encoder half of HumanSignalAutoencoder.

    Args:
        signal_shape (tuple): shape of one input sample, e.g. (1, 21, 3).
        bottleneck (int): output dimension of the latent vector.
    """

    def __init__(self, signal_shape, bottleneck):
        super().__init__()
        input_dim = 1
        for s in signal_shape:
            input_dim *= s
        body_width = (input_dim + bottleneck) // 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, body_width),
            nn.ELU(),
            nn.Linear(body_width, body_width),
            nn.ELU(),
            nn.Linear(body_width, body_width),
            nn.ELU(),
            nn.Linear(body_width, bottleneck),
        )

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


class HumanSignalDecoder(nn.Module):
    """
    Decoder half of HumanSignalAutoencoder.

    Args:
        signal_shape (tuple): target output shape, e.g. (1, 21, 3).
        bottleneck (int): input dimension of the latent vector.
    """

    def __init__(self, signal_shape, bottleneck):
        super().__init__()
        output_dim = 1
        for s in signal_shape:
            output_dim *= s
        self.signal_shape = signal_shape
        body_width = (output_dim + bottleneck) // 2

        self.net = nn.Sequential(
            nn.Linear(bottleneck, body_width),
            nn.ELU(),
            nn.Linear(body_width, body_width),
            nn.ELU(),
            nn.Linear(body_width, body_width),
            nn.ELU(),
            nn.Linear(body_width, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.shape[0], *self.signal_shape)


class HumanSignalAutoencoder(nn.Module):
    """
    Symmetric autoencoder for arbitrary-shaped human signal tensors.

    Args:
        signal_shape (tuple): shape of one input sample, e.g. (1, 21, 3).
        bottleneck (int): latent space dimension.
    """

    def __init__(self, signal_shape=(1, 21, 3), bottleneck=32):
        super().__init__()
        self.encoder = HumanSignalEncoder(signal_shape, bottleneck)
        self.decoder = HumanSignalDecoder(signal_shape, bottleneck)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
