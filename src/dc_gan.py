import torch
from torch import nn


# https://arxiv.org/pdf/1511.06434v2.pdf

class DCGenerator(nn.Module):
    def __init__(self, noize_size: int, hidden_channels: int, image_channels: int):
        super(DCGenerator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=noize_size, out_channels=hidden_channels * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.ReLU())
        self.hidden_layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channels * 8, out_channels=hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.ReLU())
        self.hidden_layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channels * 4, out_channels=hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU())
        self.hidden_layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channels * 2, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=hidden_channels, out_channels=image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x


class DCDiscriminator(nn.Module):
    def __init__(self, image_channels: int, hidden_channels: int):
        super(DCDiscriminator, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU())
        self.hidden_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.LeakyReLU())
        self.hidden_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels * 2, out_channels=hidden_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 4),
            nn.LeakyReLU())
        self.hidden_layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels * 4, out_channels=hidden_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_channels * 8),
            nn.LeakyReLU())
        self.output_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels * 8, out_channels=1, kernel_size=4, stride=2),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.output_layer(x)
        return x.reshape(-1, 1)
