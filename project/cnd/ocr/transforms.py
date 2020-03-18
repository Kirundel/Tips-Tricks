import torch
import numpy as np
import cv2
from random import randint
from imutils import rotate


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
        angle = randint(-self.rotation_angle, self.rotation_angle)
        return rotate(image, angle)


class RandomCrop:
    def __init__(self, crop_percent):
        self.crop_ratio = crop_percent / 100

    def __call__(self, image):
        tmp_x = int(self.crop_ratio * image.shape[0])
        x_l = randint(0, tmp_x)
        x_r = randint(image.shape[0] - tmp_x, image.shape[0])
        tmp_y = int(self.crop_ratio * image.shape[1])
        y_l = randint(0, tmp_y)
        y_r = randint(image.shape[1] - tmp_y, image.shape[1])
        return image[x_l:x_r, y_l:y_r]

class OneChannel:
    def __call__(self, image):
        arr = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3.0
        return arr
        #edgesx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        #edgesy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        #edges = np.linalg.norm([edgesx, edgesy], axis=0)
        #edges /= np.max(edges)
        #return edges


class TorchTransform:
    def __call__(self, image):
        return np.array([image])


def get_transforms(device):
    return [
        ToTypeTransform(np.float32),
        ImageNormalization(),
        RandomCrop(7),
        ScaleTransform((80, 32)),
        #ScaleTransform((160, 64)),
        OneChannel(),
        RandomRotate(10),
        TorchTransform()
        #ToTensor(device)
    ]
