import json
from typing import List, Dict
import numpy as np


class DataGenerator:
    """
     1. Get a clean data
     2. Put it in clean_data_array
     3. Check how much data will be generated, name it Amount
     4. External loop for Amount: take random item from clean_data_array
     5. Get a noise_value random values in range of item length, that are not the same
     6. Internal loop for random values: in item data random_value_index make inner value opposite
     7. Add item into result array
    """
    def __init__(self, noise_level: int = 3) -> None:
        self.clean_data = None
        self.clean_data_path = 'clean_data.json'
        if noise_level >= 0:
            self.noise_level = noise_level
        else:
            raise AttributeError
        self.train_data: Dict[str, list] = {}

    def get_train_data(self, data_amount: int) -> (np.array, np.array):
        clean_data, labels = self.__get_clean_data(data_amount)
        noised_data = self.__get_noised_data(clean_data)
        return (np.array(noised_data), np.array(labels))

    def get_test_data(self, data_amount: int) -> (np.array, np.array):
        if data_amount > 10:
            data_amount -= 10
        noised_data, labels = self.get_train_data(data_amount)
        clean_data_array = np.array([self.clean_data[key] for key in self.clean_data])
        noised_data = np.concatenate((noised_data, clean_data_array))
        clean_labels_array = np.array([int(key) for key in self.clean_data])
        noised_labels = np.concatenate((labels, clean_labels_array))
        to_shuffle = np.column_stack((noised_data, noised_labels))
        np.random.shuffle(to_shuffle)
        noised_data, noised_labels = np.split(to_shuffle, [-1], 1)
        noised_data = np.reshape(noised_data, (len(noised_data), noised_data.size // len(noised_data)))
        noised_labels = np.reshape(noised_labels, len(noised_labels))
        return (noised_data, noised_labels)

    def __get_noised_data(self, data: List[List[int]]) -> List[List[int]]:
        length = len(data[0])
        for item in data:
            rands = np.random.randint(length, size=self.noise_level)
            for rand in rands:
                # fast inverse
                item[rand] = 1 - item[rand]
        a = data
        return a

    def __get_clean_data(self, amount: int) -> (List[List[int]], List[int]):
        if not self.clean_data:
            with open(self.clean_data_path) as f:
                self.clean_data = json.load(f)
        data, labels = [], []
        length = len(self.clean_data)
        for _ in range(amount):
            rand = np.random.randint(length)
            data.append(self.clean_data[str(rand)].copy())
            labels.append(rand)
        return (data, labels)

