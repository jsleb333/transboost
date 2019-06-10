try:
    from transboost.datasets.datasets import ImageDataset, MNISTDataset, CIFAR10Dataset, get_train_valid_test_bank
    import datasets
except ModuleNotFoundError:
    from .datasets import ImageDataset, MNISTDataset, CIFAR10Dataset, get_train_valid_test_bank
