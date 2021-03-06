import cv2
import numpy as np
from torch.utils.data import Dataset
from os import listdir
from os.path import dirname
from random import random, seed
from re import compile

pattern = compile(r'[A-Z]\d{3}[A-Z]{2} \d{2,3}')
random_borders = [0.7, 0.8, 1.000000000000000001]


class CommonDataset(Dataset):
    def __init__(self, transforms=None, cached=True, random_seed='132131321'):
        seed(random_seed)
        self.files_path = dirname(__file__) + '/CropNumbers/Numbase/'
        data = [[], [], []]
        nb = 0
        for picture in listdir(self.files_path):
            if pattern.match(picture) is not None:
                rnd_val = random()
                for i in range(len(random_borders)):
                    if random_borders[i] >= rnd_val:
                        nb += 1
                        data[i].append((picture, '/CropNumbers/Numbase/' + picture))
                        break

        print('nb', nb)

        neg = 0

        self.files_path = dirname(__file__) + '/CropNumbers/Negative/'
        for picture in listdir(self.files_path)[:750]:
            rnd_val = random()
            for i in range(len(random_borders)):
                if random_borders[i] >= rnd_val:
                    neg += 1
                    data[i].append(('', '/CropNumbers/Negative/' + picture))
                    break

        print('neg', neg)

        gen = 0

        self.files_path = dirname(__file__) + '/Generated/'
        for picture in listdir(self.files_path):
            rnd_val = random()
            for i in range(len(random_borders)):
                if random_borders[i] >= rnd_val:
                    gen += 1
                    data[i].append((picture, '/Generated/' + picture))
                    break

        print('gen', gen)

        self.Train = CropSubDataset(dirname(__file__), data[0], transforms, cached)
        self.Validate = CropSubDataset(dirname(__file__), data[1], transforms, cached)
        self.Test = CropSubDataset(dirname(__file__), data[2], transforms, cached)


class CropDataset(Dataset):
    def __init__(self, transforms=None, cached=True, random_seed='132131321'):
        seed(random_seed)
        self.files_path = dirname(__file__) + '/CropNumbers/Numbase/'
        data = [[], [], []]
        for picture in listdir(self.files_path):
            if pattern.match(picture) is not None:
                rnd_val = random()
                for i in range(len(random_borders)):
                    if random_borders[i] >= rnd_val:
                        data[i].append((picture, picture))
                        break
        self.Train = CropSubDataset(self.files_path, data[0], transforms, cached)
        self.Validate = CropSubDataset(self.files_path, data[1], transforms, cached)
        self.Test = CropSubDataset(self.files_path, data[2], transforms, cached)
        
        
class GeneratedDataset(Dataset):
    def __init__(self, transforms=None, cached=True, random_seed='132131321'):
        seed(random_seed)
        self.files_path = dirname(__file__) + '/Generated/'
        data = [[], [], []]
        for picture in listdir(self.files_path):
            rnd_val = random()
            for i in range(len(random_borders)):
                if random_borders[i] >= rnd_val:
                    data[i].append((picture, picture))
                    break
        self.Train = CropSubDataset(self.files_path, data[0], transforms, cached)
        self.Validate = CropSubDataset(self.files_path, data[1], transforms, cached)
        self.Test = CropSubDataset(self.files_path, data[2], transforms, cached)
        

class CropSubDataset(Dataset):
    def __init__(self, files_path, data, transforms, cached):
        self.files_path = files_path
        self.data = data
        self.transforms = transforms
        self.cached = cached
        if cached:
            self.cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.cached and (idx in self.cache):
            return self.cache[idx]

        path = self.files_path + self.data[idx][1]
        X = cv2.imread(path)
        y = self.data[idx][0][:6]

        if self.transforms is not None:
            for transform in self.transforms:
                X = transform(X)

        result = (X, y)

        if self.cached:
            self.cache[idx] = result

        return result