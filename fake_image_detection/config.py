from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """Base configuration for fine-tuning a vision transformer"""

    # dataset
    data_root_dir: Path = Path("/Users/ashapatel/Documents/projects/fake-image-detection/data")
    batch_size: int = 64
    target_image_size: tuple[int, int] = (224, 224)
    prob_horizontal_flip: float = 0.25
    prob_jpeg_compress: float = 0.5
    prob_blur: float = 0.05
    prob_random_crop: float = 0.2
    jpeg_quality_range: tuple[int, int] = (20,100)
    gaussian_blur_kernel_size: int = 7

    train_split: float = 0.6
    # note that the validation & test set splits are (1-train_split)//2 each (defined in dataset.py)
    # consider making this configurable

    pretrained_model_name: str = "openai/clip-vit-base-patch16"

    # model
    embedding_size: int = 768
    num_classes: int = 2