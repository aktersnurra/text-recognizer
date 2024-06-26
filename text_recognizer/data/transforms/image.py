import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor


class ImageStem:
    def __init__(self) -> None:
        self.pil_transform = T.Compose([])
        self.pil_to_tensor = T.ToTensor()
        self.torch_transform = torch.nn.Sequential()

    def __call__(self, img: Image) -> Tensor:
        img = self.pil_transform(img)
        img = self.pil_to_tensor(img)
        with torch.no_grad():
            img = self.torch_transform(img)
        return img
