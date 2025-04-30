import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset



class LandmarkTrainDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform


        unique_landmarks = self.data_frame['landmark_id'].unique()
        self.landmark_id_to_index = {landmark_id: idx for idx, landmark_id in enumerate(unique_landmarks)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_id = self.data_frame.iloc[idx, 0]
        label = self.data_frame.iloc[idx, 1]


        label = self.landmark_id_to_index[label]

        img_path = os.path.join(self.root_dir, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class LandmarkTestDataset(Dataset):
    def __init__(self, image_ids, root_dir, transform=None):
        self.image_ids = image_ids
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]


        img_path = os.path.join(self.root_dir, img_id[0], img_id[1], img_id[2], f"{img_id}.jpg")

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, img_id
