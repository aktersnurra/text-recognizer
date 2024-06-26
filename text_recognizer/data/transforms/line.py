import random
from typing import Any, Dict

import torchvision.transforms as T
from PIL import Image

import text_recognizer.metadata.iam_lines as metadata
from text_recognizer.data.transforms.image import ImageStem


class LineStem(ImageStem):
    """A stem for handling images containing a line of text."""

    def __init__(
        self,
        augment: bool = False,
        color_jitter_kwargs: Dict[str, Any] = None,
        random_affine_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__()
        if color_jitter_kwargs is None:
            color_jitter_kwargs = {"brightness": (0.5, 1)}
        if random_affine_kwargs is None:
            random_affine_kwargs = {
                "degrees": 3,
                "translate": (0, 0.05),
                "scale": (0.4, 1.1),
                "shear": (-40, 50),
                "interpolation": T.InterpolationMode.BILINEAR,
                "fill": 0,
            }

        if augment:
            self.pil_transforms = T.Compose(
                [
                    T.ColorJitter(**color_jitter_kwargs),
                    T.RandomAffine(**random_affine_kwargs),
                ]
            )


class IamLinesStem(ImageStem):
    """A stem for handling images containing lines of text from the IAMLines dataset."""

    def __init__(
        self,
        augment: bool = False,
        color_jitter_kwargs: Dict[str, Any] = None,
        random_affine_kwargs: Dict[str, Any] = None,
    ) -> None:
        super().__init__()

        def embed_crop(crop, augment=augment):
            # crop is PIL.image of dtype="L" (so values range from 0 -> 255)
            image = Image.new("L", (metadata.IMAGE_WIDTH, metadata.IMAGE_HEIGHT))

            # Resize crop
            crop_width, crop_height = crop.size
            new_crop_height = metadata.IMAGE_HEIGHT
            new_crop_width = int(new_crop_height * (crop_width / crop_height))
            if augment:
                # Add random stretching
                new_crop_width = int(new_crop_width * random.uniform(0.9, 1.1))
                new_crop_width = min(new_crop_width, metadata.IMAGE_WIDTH)
            crop_resized = crop.resize(
                (new_crop_width, new_crop_height), resample=Image.BILINEAR
            )

            # Embed in the image
            x = min(metadata.CHAR_WIDTH, metadata.IMAGE_WIDTH - new_crop_width)
            y = metadata.IMAGE_HEIGHT - new_crop_height

            image.paste(crop_resized, (x, y))

            return image

        if color_jitter_kwargs is None:
            color_jitter_kwargs = {"brightness": (0.8, 1.6)}
        if random_affine_kwargs is None:
            random_affine_kwargs = {
                "degrees": 1,
                "shear": (-30, 20),
                "interpolation": T.InterpolationMode.BILINEAR,
                "fill": 0,
            }

        pil_transform_list = [T.Lambda(embed_crop)]
        if augment:
            pil_transform_list += [
                T.ColorJitter(**color_jitter_kwargs),
                T.RandomAffine(**random_affine_kwargs),
            ]
        self.pil_transform = T.Compose(pil_transform_list)
