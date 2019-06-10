try:
    from transboost.datasets.datasets import ImageDataset, MNISTDataset, CIFAR10Dataset
except ModuleNotFoundError:
    from .datasets import ImageDataset, MNISTDataset, CIFAR10Dataset
