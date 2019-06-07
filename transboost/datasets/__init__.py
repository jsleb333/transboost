try:
    from transboost.datasets.datasets import ImageDataset, MNISTDataset, CIFAR10Dataset
    import datasets
except ModuleNotFoundError:
    from .datasets import ImageDataset, MNISTDataset, CIFAR10Dataset
