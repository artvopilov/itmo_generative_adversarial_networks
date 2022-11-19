import os
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image

from src.image_utils import ImageUtils
from src.utils import Utils


class CGanTrainer:
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
            num_classes: int,
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
            for i, (real_images, real_labels) in enumerate(dataset_loader):
                real_images = ImageUtils.normalize(real_images).reshape(real_images.shape[0], *input_shape).to(self._device)
                real_labels = real_labels.to(self._device)

                d_loss, real_predictions, fake_predictions = self._train_discriminator(
                    discriminator,
                    generator,
                    d_optimizer,
                    loss_function,
                    num_classes,
                    real_images,
                    real_labels,
                    real_images.shape[0],
                    noize_shape)

                g_loss, fake_images = self._train_generator(
                    discriminator,
                    generator,
                    g_optimizer,
                    loss_function,
                    num_classes,
                    real_images.shape[0],
                    noize_shape)

                if (i + 1) % 100 == 0:
                    d_losses.append(d_loss.item())
                    g_losses.append(g_loss.item())
                    real_scores.append(real_predictions.mean().item())
                    fake_scores.append(fake_predictions.mean().item())
                    Utils.log_step(epoch, epochs, i + 1, total_step, d_loss, g_loss, real_predictions, fake_predictions)
            self._save_fake_images(generator, num_classes, noize_shape, image_shape, epoch + 1, images_dir)

        Utils.log_generated_images(images_dir, epochs)
        Utils.log_losses(d_losses, g_losses)
        Utils.log_scores(real_scores, fake_scores)

    def _train_discriminator(
            self,
            discriminator: nn.Module,
            generator: nn.Module,
            d_optimizer: optim.Optimizer,
            loss_function: nn.Module,
            num_classes: int,
            real_images: torch.Tensor,
            real_labels: torch.Tensor,
            batch_size: int,
            noize_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d_optimizer.zero_grad()

        real_scores = torch.ones(batch_size, 1).to(self._device)
        fake_scores = torch.zeros(batch_size, 1).to(self._device)

        real_predictions = discriminator(real_images, real_labels)
        noize = torch.randn(batch_size, *noize_shape).to(self._device)
        fake_labels = torch.randint(0, num_classes, (batch_size,)).to(self._device)
        fake_images = generator(noize, fake_labels)
        fake_predictions = discriminator(fake_images, fake_labels)

        real_images_loss = loss_function(real_predictions, real_scores)
        fake_images_loss = loss_function(fake_predictions, fake_scores)
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
            num_classes: int,
            batch_size: int,
            noize_shape: Tuple[int, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g_optimizer.zero_grad()

        fake_scores = torch.ones(batch_size, 1).to(self._device)

        noize = torch.randn(batch_size, *noize_shape).to(self._device)
        fake_labels = torch.randint(0, num_classes, (batch_size,)).to(self._device)
        fake_images = generator(noize, fake_labels)
        fake_predictions = discriminator(fake_images, fake_labels)

        loss = loss_function(fake_predictions, fake_scores)
        loss.backward()
        g_optimizer.step()

        return loss, fake_images

    def _save_fake_images(
            self,
            generator: nn.Module,
            num_classes: int,
            noize_shape: Tuple[int, ...],
            image_shape: Tuple[int, ...],
            epoch: int,
            images_dir: str
    ) -> None:
        noize = torch.randn(num_classes * 10, *noize_shape).to(self._device)
        fake_labels = torch.IntTensor([label for _ in range(10) for label in range(num_classes)])
        fake_images = generator(noize, fake_labels).reshape(num_classes * 10, *image_shape)

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        file_path = Utils.get_image_path(images_dir, epoch)
        print("Saving", file_path)
        save_image(ImageUtils.denormalize(fake_images), file_path, nrow=10)
