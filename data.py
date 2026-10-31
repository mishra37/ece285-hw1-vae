"""Loading the Kaggle chest X-ray (pneumonia) dataset.

Each image is converted to grayscale, resized so its shorter side is `img_size`,
center-cropped to a square and scaled to [0, 1]. Decoding ~5k JPEGs is slow, so each
split is saved as a single tensor file the first time and reused after that.

Expects the images in data/chest_xray/{train,test}/{NORMAL,PNEUMONIA}/*.jpeg. The class
labels are loaded too, but the VAE never uses them.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision import transforms
from tqdm import tqdm

DATA_ROOT = Path(__file__).parent / "data" / "chest_xray"
CACHE_DIR = Path(__file__).parent / "data" / "cache"
CLASSES = ["NORMAL", "PNEUMONIA"]


def _load_split(split, img_size):
    preprocess = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    image_files = []
    for label, class_name in enumerate(CLASSES):
        for path in sorted((DATA_ROOT / split / class_name).glob("*.jpeg")):
            image_files.append((path, label))

    images, labels = [], []
    for path, label in tqdm(image_files, desc=f"loading {split}"):
        with Image.open(path) as image:
            images.append(preprocess(image))
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def get_dataset(split, img_size=64):
    """Return a TensorDataset of (image, label) pairs for 'train' or 'test'."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{split}_{img_size}.pt"
    if cache_file.exists():
        images, labels = torch.load(cache_file)
    else:
        images, labels = _load_split(split, img_size)
        torch.save((images, labels), cache_file)
    return TensorDataset(images, labels)
