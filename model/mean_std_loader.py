import io
import os
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch


class StatsFromDataSet(Dataset):
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size

    def init_large_loader(self, w, h, shuffle=False, num_workers=0):
        transform_img = transforms.Compose(
            [
                transforms.Resize((w, h)),
                transforms.CenterCrop((w, h)),
                transforms.ToTensor(),
                # here do not use transforms.Normalize(mean, std)
            ]
        )

        image_data = torchvision.datasets.ImageFolder(
            root=self.data_path, transform=transform_img
        )

        image_data_loader = DataLoader(
            image_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = os.path.join(self.root_dir, self.class_name.iloc[idx, 0])
            image = io.imread(img_name)
            class_name = self.class_name.iloc[idx, 1:]
            class_name = np.array([class_name])
            class_name = class_name.astype("float").reshape(-1, 2)
            sample = {"image": image, "class": class_name}

            if self.transform:
                sample = self.transform(sample)

        return image_data_loader

    def batch_mean_and_std(self, loader):
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)
        print(
            "Calculating mean and std for images on folder '" + self.data_path + "'..."
        )
        for images, _ in loader:
            b, c, h, w = images.shape
            nb_pixels = b * h * w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images**2, dim=[0, 2, 3])
            fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
            cnt += nb_pixels

        mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment**2)
        return mean, std
