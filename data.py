import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.data import Subset, Dataset


class TorchvisionDatasetTimm(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, index):
        img, target = self._dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self._dataset)


def create_oxford_pets_dataset(root="data", is_training=True, n_img_per_class=10, random_seed=0):
    if is_training:
        dataset = torchvision.datasets.OxfordIIITPet(root, split="trainval", download=True)

        # Make a subset of the training dataset with N images per class

        np.random.seed(random_seed)

        targets = [cls for _, cls in dataset]

        subset = []
        for cls in range(len(dataset.classes)):
            indices = np.where(np.array(targets) == cls)[0]
            subset.extend(np.random.choice(indices, n_img_per_class, replace=False))

        dataset = Subset(dataset, subset)
    else:
        dataset = torchvision.datasets.OxfordIIITPet(root, split="test")

    return TorchvisionDatasetTimm(dataset)


def create_flowers102_dataset(root="data", is_training=True, n_img_per_class=10, random_seed=0):
    if is_training:
        dataset = torchvision.datasets.Flowers102(root, split="train", download=True)

        # Make a subset of the training dataset with N images per class

        np.random.seed(random_seed)

        targets = [cls for _, cls in dataset]
        classes = set(targets)

        subset = []
        for cls in classes:
            indices = np.where(np.array(targets) == cls)[0]
            subset.extend(np.random.choice(indices, n_img_per_class, replace=False))

        dataset = Subset(dataset, subset)
    else:
        dataset = torchvision.datasets.Flowers102(root, split="val")

    return TorchvisionDatasetTimm(dataset)



"""
class ParserImageName(Parser):
    def __init__(self, root, class_to_idx=None):
        super().__init__()

        self.root = Path(root)
        self.samples = list(self.root.glob("*.jpg"))

        if class_to_idx:
            self.class_to_idx = class_to_idx
        else:
            classes = sorted(
                set([self.__extract_label_from_path(p) for p in self.samples]),
                key=lambda s: s.lower(),
            )
            self.class_to_idx = {c: idx for idx, c in enumerate(classes)}

    def __extract_label_from_path(self, path):
        return "_".join(path.parts[-1].split("_")[0:-1])

    def __getitem__(self, index):
        path = self.samples[index]
        target = self.class_to_idx[self.__extract_label_from_path(path)]
        return open(path, "rb"), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = filename.parts[-1]
        elif not absolute:
            filename = filename.absolute()
        return filename
"""