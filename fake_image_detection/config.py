from dataclasses import dataclass

@dataclass
class Config:
    """Base configuration for fine-tuning a vision transformer"""

    # dataset
    data_root_dir: Path = Path("/Users/ashapatel/Documents/projects/fake-image-detection/data")
    batch_size: int = 64
    target_image_size: int = (768, 768)
    prob_horizontal_flip: float = 0.25
    prob_jpeg_compress: float = 0.5
    prob_blur: float = 0.05
    prob_random_crop: float = 0.2
    jpeg_quality_range: list[int, int] = [20,100]
    gaussian_blur_kernel_size = 7

    train_split: float = 0.6
    # note that the val tests splits are (1-train_split)//2 each
    # consider making this configurable

    pretrained_model_name: str = "openai/clip-vit-base-patch16"