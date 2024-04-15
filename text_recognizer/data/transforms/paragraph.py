"""Iam paragraph stem class."""
import torchvision.transforms as T

import text_recognizer.metadata.iam_paragraphs as metadata
from text_recognizer.data.transforms.image import ImageStem

IMAGE_HEIGHT, IMAGE_WIDTH = metadata.IMAGE_HEIGHT, metadata.IMAGE_WIDTH
IMAGE_SHAPE = metadata.IMAGE_SHAPE

MAX_LABEL_LENGTH = metadata.MAX_LABEL_LENGTH


class ParagraphStem(ImageStem):
    """A stem for handling images that contain a paragraph of text."""

    def __init__(
        self,
        augment=False,
        color_jitter_kwargs=None,
        random_affine_kwargs=None,
        random_perspective_kwargs=None,
        gaussian_blur_kwargs=None,
        sharpness_kwargs=None,
    ):
        super().__init__()

        if not augment:
            self.pil_transform = T.Compose([T.CenterCrop(IMAGE_SHAPE)])
        else:
            if color_jitter_kwargs is None:
                color_jitter_kwargs = {"brightness": 0.4, "contrast": 0.4}
            if random_affine_kwargs is None:
                random_affine_kwargs = {
                    "degrees": 3,
                    "shear": 6,
                    "scale": (0.95, 1),
                    "interpolation": T.InterpolationMode.BILINEAR,
                }
            if random_perspective_kwargs is None:
                random_perspective_kwargs = {
                    "distortion_scale": 0.2,
                    "p": 0.5,
                    "interpolation": T.InterpolationMode.BILINEAR,
                }
            if gaussian_blur_kwargs is None:
                gaussian_blur_kwargs = {"kernel_size": (3, 3), "sigma": (0.1, 1.0)}
            if sharpness_kwargs is None:
                sharpness_kwargs = {"sharpness_factor": 2, "p": 0.5}

            self.pil_transform = T.Compose(
                [
                    T.ColorJitter(**color_jitter_kwargs),
                    T.RandomCrop(
                        size=IMAGE_SHAPE,
                        padding=None,
                        pad_if_needed=True,
                        fill=0,
                        padding_mode="constant",
                    ),
                    T.RandomAffine(**random_affine_kwargs),
                    T.RandomPerspective(**random_perspective_kwargs),
                    T.GaussianBlur(**gaussian_blur_kwargs),
                    T.RandomAdjustSharpness(**sharpness_kwargs),
                ]
            )
