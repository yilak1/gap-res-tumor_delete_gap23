from data_handler import get_data
from torch.utils.data import Dataset
import torch
DATASET_FOLDER = "dataset/dataset6-4-1"
dataset = "./dataset/dataset6-4-1"

X_train, y_train, X_test, y_test = get_data(dataset)

print("y_train", y_train)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
# X_valid = X_valid / 255
X_test = X_test / 255
y_train = y_train.astype('long')
y_test = y_test.astype('long')
X_train = X_train.transpose((0, 3, 1, 2))
X_test = X_test.transpose((0, 3, 1, 2))

class TumorTrain(Dataset):
    def __init__(self, transform=None):
        self.images_train, self.labels_train = X_train, y_train
        self.transform = transform

    def __len__(self):
        return len(self.labels_train)

    def __getitem__(self, item):
        images_train = self.images_train

        images_train, labels_train = torch.from_numpy(images_train), torch.from_numpy(self.labels_train)
        return images_train[item], labels_train[item]


class TumorTest(Dataset):
    def __init__(self):
        self.images_test, self.labels_test = X_test, y_test

    def __len__(self):
        return len(self.images_test)

    def __getitem__(self, item):
        images_test = self.images_test
        images_test, labels_test = torch.from_numpy(images_test), torch.from_numpy(self.labels_test)
        return images_test[item], self.labels_test[item]
