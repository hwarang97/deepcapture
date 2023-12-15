import os

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from PIL import UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def split_dataset(folder_path, label, val_size=0.1, test_size=0.1):
    all_images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith((".png", ".jpg", ".jpeg", ".gif"))
    ]

    # split dataset
    train_data, temp_data = train_test_split(
        all_images, test_size=val_size + test_size, random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=test_size / (val_size + test_size), random_state=42
    )

    # assign label
    train_data = [(path, label) for path in train_data]
    val_data = [(path, label) for path in val_data]
    test_data = [(path, label) for path in test_data]

    return train_data, val_data, test_data


class ImageDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = [item for item in data_list if self._is_valid_image(item[0])]
        self.transform = transform

    def _is_valid_image(self, img_path):
        try:
            with Image.open(img_path) as image:
                image = np.array(image)
            return True
        except (OSError, UnidentifiedImageError):
            return False

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]

        with Image.open(img_path) as image:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image = np.array(image)

        # if grayscale, convert to RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def get_loaders(train_data, val_data, test_data, batch_size=32):
    train_transform = A.Compose(
        [
            A.RandomResizedCrop(height=32, width=32, scale=(0.8, 0.8)),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.Resize(32, 32),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )

    train_dataset = ImageDataset(train_data, transform=train_transform)
    val_dataset = ImageDataset(val_data, transform=test_transform)
    test_dataset = ImageDataset(test_data, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader
