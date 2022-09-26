import text_recognizer.metadata.emnist as emnist
import text_recognizer.metadata.shared as shared

PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "iam_lines"

IMAGE_SCALE_FACTOR = 2

CHAR_WIDTH = emnist.INPUT_SHAPE[0] // IMAGE_SCALE_FACTOR  # rough estimate
IMAGE_HEIGHT = 112 // IMAGE_SCALE_FACTOR
IMAGE_WIDTH = 3072 // IMAGE_SCALE_FACTOR  # rounding up IAMLines empirical maximum width

SEED = 4711
DIMS = (1, IMAGE_HEIGHT, IMAGE_WIDTH)
OUTPUT_DIMS = (89, 1)
IMAGE_HEIGHT = 56
IMAGE_WIDTH = 1024
MAX_LABEL_LENGTH = 89
MAX_WORD_PIECE_LENGTH = 72

MAPPING = emnist.MAPPING
