import torch
import numpy as np
from transforms import get_transforms_without_augmentations
from converter import strLabelConverter
from model import CRNN
from common import preds_converter, alphabet, model_parameters


class Predictor:
    def __init__(self, device="cpu"):
        self.model = CRNN(
            **model_parameters
        )
        model_data = torch.load('logs/checkpoints/best_full.pth')
        self.model.load_state_dict(model_data['model_state_dict'])
        self.transforms = get_transforms_without_augmentations(device)
        self.converter = strLabelConverter(alphabet)

    def predict(self, images):
        if type(images) != list:
            images = [images]

        for i in range(len(images)):
            for transform in self.transforms:
                images[i] = transform(images[i])

        arr = torch.tensor(np.stack(images))

        pred = self.model(arr)
        text, _ = preds_converter(self.converter, pred, len(images))
        return text