import os
import dataset.utils as U


class BaseDataset:
    def __init__(
        self, name: str, dataset_dir: str, num_workers: str, modal: str
    ) -> None:
        self.dataset = name
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.label_type = "failure_type"

        self.__y__ = {"failure_type": [], "root_cause": []}
        self.__X__ = []

        self.services = []
        self.instances = []
        self.failures = []

        self.data_path = os.path.join("tmp", f"{self.dataset}_{modal}_tmp.json")
        U.check(self.data_path)

    def set_label_type(self, label_type: str):
        if label_type not in self.__y__:
            raise ValueError(f"Invalid label type: {label_type}")
        self.label_type = label_type

    @property
    def X(self):
        return self.__X__

    @property
    def y(self):
        return self.__y__[self.label_type]

    def load(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        return self.__X__[index], self.__y__[self.label_type][index]

    def __len__(self):
        return len(self.__X__)

    def data_argument(self, X, y):
        return X, y
