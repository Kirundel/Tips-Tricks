import torch
import numpy as np
import cv2
from random import uniform


class ImageNormalization:
    def __call__(self, image):
        return image / 255.


class ScaleTransform:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, image):
        return cv2.resize(image, self.shape)


class ToTypeTransform:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image):
        if self.dtype == image.dtype:
            return image
        return image.astype(self.dtype)


class ToTensor:
    def __init__(self, device):
        self.device = device

    def __call__(self, image):
        return torch.from_numpy(image).to(self.device)


class RandomRotate:
    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def __call__(self, image):
        angle = int(uniform(-self.rotation_angle, self.rotation_angle) + 0.5)
        return cv2.rotate(image, angle)


class TorchTransform:
    def __call__(self, image):
        return np.transpose(image, (2, 0, 1))


def get_transforms(device):
    return [
        ToTypeTransform(np.float32),
        ImageNormalization(),
        ScaleTransform((80, 32)),
        TorchTransform()
        #RandomRotate(10),
        #ToTensor(device)
    ]
