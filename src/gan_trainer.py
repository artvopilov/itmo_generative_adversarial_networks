import os
from typing import Tuple, List

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from src.image_utils import ImageUtils


class GanTrainer:
    _device: str

    def __init__(self, device: str) -> None:
        self._device = device

    def train(
            self,
            dataset: Dataset,
            generator: nn.Module,
            discriminator: nn.Module,
            g_optimizer: optim.Optimizer,
            d_optimizer: optim.Optimizer,
            loss_function: nn.Module,
            batch_size: int,
            noize_shape: Tuple[int, ...],
            input_shape: Tuple[int, ...],
            image_shape: Tuple[int, ...],
            epochs: int,
            images_dir: str
    ) -> None:
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        generator.to(self._device)
        discriminator.to(self._device)

        total_step = len(dataset_loader)
        g_losses, d_losses, real_scores, fake_scores = [], [], [], []
        for epoch in range(epochs):
            for i, (real_images, _) in enumerate(dataset_loader):
                real_images = ImageUtils.normalize(real_images).reshape(real_images.shape[0], *input_shape).to(self._device)

                d_loss, real_predictions, fake_predictions = self._train_discriminator(
                    discriminator,
                    generator,
                    d_optimizer,
                    loss_function,
                    real_images,
                    real_images.shape[0],
                    noize_shape)

                g_loss, fake_images = self._train_generator(
                    discriminator,
                    generator,
                    g_optimizer,
                    loss_function,
                    real_images.shape[0],
                    noize_shape)

                if (i + 1) % 100 == 0:
                    d_losses.append(d_loss.item())
                    g_losses.append(g_loss.item())
                    real_scores.append(real_predictions.mean().item())
                    fake_scores.append(fake_predictions.mean().item())
                    self._log_step(epoch, epochs, i + 1, total_step, d_loss, g_loss, real_predictions, fake_predictions)
            self._save_fake_images(generator, batch_size, noize_shape, image_shape, epoch + 1, images_dir)

        self._log_generated_images(images_dir, epochs)
        self._log_losses(d_losses, g_losses)
        self._log_scores(real_scores, fake_scores)

    def _train_discriminator(
            self,
            discriminator: nn.Module,
            generator: nn.Module,
            d_optimizer: optim.Optimizer,
            loss_function: nn.Module,
            real_images: torch.Tensor,
            batch_size: int,
            noize_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_optimizer.zero_grad()

        real_labels = torch.ones(batch_size, 1).to(self._device)
        fake_labels = torch.zeros(batch_size, 1).to(self._device)

        real_predictions = discriminator(real_images)
        noize = torch.randn(batch_size, *noize_shape).to(self._device)
        fake_images = generator(noize)
        fake_predictions = discriminator(fake_images)

        real_images_loss = loss_function(real_predictions, real_labels)
        fake_images_loss = loss_function(fake_predictions, fake_labels)
        loss = real_images_loss + fake_images_loss
        loss.backward()
        d_optimizer.step()

        return loss, real_predictions, fake_predictions

    def _train_generator(
            self,
            discriminator: nn.Module,
            generator: nn.Module,
            g_optimizer: optim.Optimizer,
            loss_function: nn.Module,
            batch_size: int,
            noize_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g_optimizer.zero_grad()

        labels = torch.ones(batch_size, 1).to(self._device)
        noize = torch.randn(batch_size, *noize_shape).to(self._device)
        images = generator(noize)
        predictions = discriminator(images)

        loss = loss_function(predictions, labels)
        loss.backward()
        g_optimizer.step()

        return loss, images

    @staticmethod
    def _log_step(
            epoch: int,
            epochs: int,
            step: int,
            steps: int,
            d_loss: torch.Tensor,
            g_loss: torch.Tensor,
            real_predictions: torch.Tensor,
            fake_predictions: torch.Tensor
    ) -> None:
        print("Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}".format(
            epoch, epochs, step, steps, d_loss.item(), g_loss.item(),
            real_predictions.mean().item(), fake_predictions.mean().item()))

    def _save_fake_images(
            self,
            generator: nn.Module,
            batch_size: int,
            noize_shape: Tuple[int, ...],
            image_shape: Tuple[int, ...],
            epoch: int,
            images_dir: str
    ) -> None:
        z = torch.randn(batch_size, *noize_shape).to(self._device)
        fake_images = generator(z).reshape(batch_size, *image_shape)

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        file_path = self._get_image_path(images_dir, epoch)
        print("Saving", file_path)
        save_image(ImageUtils.denormalize(fake_images), file_path, nrow=10)

    def _log_generated_images(self, images_dir: str, epoch: int) -> None:
        file_path = self._get_image_path(images_dir, epoch)
        print("Loading", file_path)
        image = Image.open(file_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    @staticmethod
    def _log_losses(d_losses: List[float], g_losses: List[float]) -> None:
        plt.plot(d_losses, "-")
        plt.plot(g_losses, "-")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["Discriminator", "Generator"])
        plt.title("Losses")
        plt.show()

    @staticmethod
    def _log_scores(real_scores: List[float], fake_scores: List[float]) -> None:
        plt.plot(real_scores, "-")
        plt.plot(fake_scores, "-")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.legend(["Real Score", "Fake score"])
        plt.title("Scores")
        plt.show()

    @staticmethod
    def _get_image_path(images_dir: str, epoch: int) -> str:
        return os.path.join(images_dir, f'fake_images_{epoch}.png')
