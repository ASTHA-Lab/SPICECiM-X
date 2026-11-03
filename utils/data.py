import torch
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets


def _to_rgb(im):
    """Module-level (picklable for DataLoader workers on Windows/spawn)."""
    return im.convert("RGB")


# NOTE: the non-augmenting transform must stay identical to
# spicecimx.io.torch_bridge.build_preprocess (used at inference time).
def _transforms(img_size, channels, augment=False):
    ops = [T.Resize(img_size)]
    if augment:
        ops += [T.RandomHorizontalFlip(), T.RandomCrop(img_size, padding=4)]
    if isinstance(channels, int) or (isinstance(channels, str) and channels != "auto"):
        ch = int(channels)
        if ch == 1:
            ops += [T.Grayscale(num_output_channels=1)]
        elif ch == 3:
            # NOT Grayscale(3): that converts colour images to grayscale.
            # convert("RGB") keeps RGB untouched and expands grayscale to 3ch.
            ops += [T.Lambda(_to_rgb)]
    ops += [T.ToTensor()]
    return T.Compose(ops)

def prepare_dataloaders(dataset_name: str, data_root: str, img_size, channels, batch_size: int, num_workers: int, val_split: float, augment: bool):
    dataset_name = dataset_name.lower()
    train_tfm = _transforms(img_size, channels, augment)
    eval_tfm = _transforms(img_size, channels, False)
    if dataset_name == "mnist":
        ds_cls, default_c = dsets.MNIST, 1
    elif dataset_name in ("fashion", "fashion_mnist", "mnist-fashion", "fashionmnist"):
        ds_cls, default_c = dsets.FashionMNIST, 1
    elif dataset_name == "cifar10":
        ds_cls, default_c = dsets.CIFAR10, 3
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Two views of the same training files: augmenting for the train subset,
    # eval transform for the held-out validation subset.
    full_train = ds_cls(root=data_root, train=True, download=True, transform=train_tfm)
    val_len = int(len(full_train) * val_split) if val_split > 0 else 0
    if val_len > 0:
        full_train_eval = ds_cls(root=data_root, train=True, download=True, transform=eval_tfm)
        perm = torch.randperm(len(full_train)).tolist()
        train_set = Subset(full_train, perm[val_len:])
        val_set = Subset(full_train_eval, perm[:val_len])
    else:
        # val_split <= 0: no hold-out; the official test split is used for
        # validation / best-checkpoint selection.
        train_set = full_train
        val_set = ds_cls(root=data_root, train=False, download=True, transform=eval_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    in_channels = default_c if channels == "auto" else int(channels)
    num_classes = len(full_train.classes)
    return train_loader, val_loader, num_classes, in_channels
