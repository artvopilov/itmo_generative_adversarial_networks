import torch
from torch import nn


# https://arxiv.org/pdf/1411.1784.pdf

class CGenerator(nn.Module):
    def __init__(self, noize_size: int, hidden_size: int, image_size: int, num_classes: int) -> None:
        super(CGenerator, self).__init__()
        self.label_embed_layer = nn.Embedding(num_classes, num_classes)
        self.input_layer = nn.Sequential(
            nn.Linear(noize_size + num_classes, hidden_size),
            nn.ReLU())
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, image_size ** 2),
            nn.Tanh())

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, self.label_embed_layer(labels)), -1)
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


class CDiscriminator(nn.Module):
    def __init__(self, image_size: int, hidden_size: int, num_classes: int) -> None:
        super(CDiscriminator, self).__init__()
        self.label_embed_layer = nn.Embedding(num_classes, num_classes)
        self.input_layer = nn.Sequential(
            nn.Linear(image_size ** 2 + num_classes, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2))
        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2))
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, self.label_embed_layer(labels)), -1)
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x
