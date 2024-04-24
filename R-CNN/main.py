import torch as tc
from torchvision.datasets import VOCDetection, CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import helpers
from debug import show, matrix_verbose
from model import CNN


def main():
    device = "cuda" if tc.cuda.is_available() else "cpu"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((227, 227), antialias=True),
            transforms.Normalize(
                mean=(0.49139968, 0.48215827, 0.44653124),
                std=(0.24703233, 0.24348505, 0.26158768),
            ),
        ]
    )

    pascal_voc2012 = VOCDetection(
        root="./data", download=False, transform=transforms.PILToTensor()
    )

    cifar10_train = CIFAR10(
        root="./data", download=False, train=True, transform=transform
    )

    train_size = int(0.8 * len(cifar10_train))
    val_size = len(cifar10_train) - train_size

    train_dataset, val_dataset = random_split(cifar10_train, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=2
    )

    test_dataset = CIFAR10(
        root="./data", download=True, train=False, transform=transform
    )

    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=2
    )

    image = next(iter(train_loader))[0]
    matrix_verbose(image)


if __name__ == "__main__":
    main()
