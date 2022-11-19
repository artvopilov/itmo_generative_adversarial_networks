import os
from typing import List

import matplotlib.pyplot as plt
import torch
from PIL import Image


class Utils:
    @staticmethod
    def log_step(
            epoch: int,
            epochs: int,
            step: int,
            steps: int,
            d_loss: torch.Tensor,
            g_loss: torch.Tensor,
            real_scores: torch.Tensor,
            fake_scores: torch.Tensor
    ) -> None:
        print("Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}".format(
            epoch, epochs, step, steps, d_loss.item(), g_loss.item(),
            real_scores.mean().item(), fake_scores.mean().item()))

    @staticmethod
    def log_generated_images(images_dir: str, epoch: int) -> None:
        file_path = Utils.get_image_path(images_dir, epoch)
        print("Loading", file_path)
        image = Image.open(file_path)
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    @staticmethod
    def log_losses(d_losses: List[float], g_losses: List[float]) -> None:
        plt.plot(d_losses, "-")
        plt.plot(g_losses, "-")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(["Discriminator", "Generator"])
        plt.title("Losses")
        plt.show()

    @staticmethod
    def log_scores(real_scores: List[float], fake_scores: List[float]) -> None:
        plt.plot(real_scores, "-")
        plt.plot(fake_scores, "-")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.legend(["Real Score", "Fake score"])
        plt.title("Scores")
        plt.show()

    @staticmethod
    def get_image_path(images_dir: str, epoch: int) -> str:
        return os.path.join(images_dir, f'fake_images_{epoch}.png')
