import torch
from torch import nn


# https://arxiv.org/pdf/1406.2661.pdf

class Generator(nn.Module):
    def __init__(self, noize_size: int, hidden_size: int, image_size: int) -> None:
        super(Generator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(noize_size, hidden_size),
            nn.ReLU())
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, image_size ** 2),
            nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_size: int, hidden_size: int) -> None:
        super(Discriminator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(image_size ** 2, hidden_size),
            nn.LeakyReLU())
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU())
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x
