import torch
import numpy as np
from cnd.ocr.transforms import get_transforms_without_augmentations
from cnd.ocr.converter import strLabelConverter
from cnd.ocr.model import CRNN
from cnd.ocr.common import preds_converter, alphabet, model_parameters
from os.path import dirname

class Predictor:
    def __init__(self, device="cpu"):
        self.model = CRNN(
            **model_parameters
        )
        model_data = torch.load(dirname(__file__) + '/logs/checkpoints/best_full.pth')
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

        if len(text) > 6:
            text = text[:6]
        elif len(text) == 0:
            text = ""

        return text