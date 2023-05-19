import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

from .constants import DATASET_BHW_SPLIT_RANDOM_SEED, DATASET_BHW_TEST_SIZE


class DatasetBHW(Dataset):
    SPLIT_RANDOM_SEED = DATASET_BHW_SPLIT_RANDOM_SEED
    TEST_SIZE = DATASET_BHW_TEST_SIZE

    def __init__(self, root, path_to_labels, train, transform=None):
        super().__init__()
        self.root = root
        self.path_to_labels = path_to_labels
        self.train = train
        self.transform = transform

        labels_data = pd.read_csv(path_to_labels)
        test = labels_data.sample(frac=self.TEST_SIZE, random_state=self.SPLIT_RANDOM_SEED)
        train = labels_data.drop(test.index)
        self.labels_data = train if self.train else test

    def __len__(self):
        return len(self.labels_data)

    def __getitem__(self, item):
        label = self.labels_data.iloc[item, 1]
        filename = self.labels_data.iloc[item, 0]
        image = Image.open(os.path.join(self.root, filename)).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label
