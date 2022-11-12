import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import Normalize, ToTensor, Resize, Compose, InterpolationMode


class ImageUtils:
    _NORMALIZATION: Normalize = Normalize(mean=(0.5,), std=(0.5,))
    _DE_NORMALIZATION: Normalize = Normalize(mean=(-1,), std=(2,))

    @staticmethod
    def read_mnist_dataset(root_dir: str, train: bool = True, download: bool = True) -> MNIST:
        return MNIST(
            root=root_dir,
            train=train,
            download=download,
            transform=ToTensor())

    @staticmethod
    def read_cifar10_dataset(root_dir: str, image_size: int, train: bool = True, download: bool = True) -> CIFAR10:
        transform = Compose([Resize([image_size, image_size], InterpolationMode.BILINEAR), ToTensor()])
        return CIFAR10(
            root=root_dir,
            train=train,
            download=download,
            transform=transform)

    @staticmethod
    def normalize(image: torch.Tensor) -> torch.Tensor:
        return ImageUtils._NORMALIZATION(image)

    @staticmethod
    def denormalize(image: torch.Tensor) -> torch.Tensor:
        return ImageUtils._DE_NORMALIZATION(image)
