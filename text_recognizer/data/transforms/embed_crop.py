"""Transforms for PyTorch datasets."""
import random

from PIL import Image


class EmbedCrop:

    IMAGE_HEIGHT = 56
    IMAGE_WIDTH = 1024

    def __init__(self, augment: bool) -> None:
        self.augment = augment

    def __call__(self, crop: Image) -> Image:
        # Crop is PIL.Image of dtype="L" (so value range is [0, 255])
        image = Image.new("L", (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))

        # Resize crop.
        crop_width, crop_height = crop.size
        new_crop_height = self.IMAGE_HEIGHT
        new_crop_width = int(new_crop_height * (crop_width / crop_height))

        if self.augment:
            # Add random stretching
            new_crop_width = int(new_crop_width * random.uniform(0.9, 1.1))
            new_crop_width = min(new_crop_width, self.IMAGE_WIDTH)
        crop_resized = crop.resize(
            (new_crop_width, new_crop_height), resample=Image.BILINEAR
        )

        # Embed in image
        x = min(28, self.IMAGE_WIDTH - new_crop_width)
        y = self.IMAGE_HEIGHT - new_crop_height
        image.paste(crop_resized, (x, y))

        return image
