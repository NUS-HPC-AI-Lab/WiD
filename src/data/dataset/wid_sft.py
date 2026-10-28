"""The paired-image SFT index used by the 40K DeCo+WiD experiment."""

import os

from src.data.dataset.image_txt import ImageText


class VerifiedImageTextSFT(ImageText):
    def __init__(self, root, resolution=512, expected_pairs=58_859):
        super().__init__(root=root, resolution=resolution)
        if resolution != 512:
            raise ValueError("The released SFT stage uses 512-pixel crops")
        if len(self.image_paths) != expected_pairs or len(self.texts) != expected_pairs:
            raise ValueError(f"Expected {expected_pairs} SFT pairs, found {len(self.image_paths)}")
        # Preserve the historical sorted index. In particular, do NOT substitute
        # the WDS pretraining resize for ImageText's torchvision Resize.
        pairs = sorted(zip(self.image_paths, self.texts), key=lambda item: item[0])
        if not all(text.strip() for _, text in pairs):
            raise ValueError("Blank SFT caption")
        self.image_paths = [path for path, _ in pairs]
        self.texts = [text for _, text in pairs]
        if not all(os.path.isfile(os.path.splitext(path)[0] + ".txt") for path in self.image_paths):
            raise ValueError("Unpaired SFT image")
