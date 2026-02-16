import os
import logging
from pathlib import Path
from transformers import CLIPProcessor
import torchvision.transforms as transforms
from fake_image_detection.config import Config
from torch.utils.data import DataLoader

log_config = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(log_config)


LABELS_DICT = {
    "real": 0,
    "synthetic": 1
}


IMAGE_FOLDER_NAMES_TO_LABELS = {
        "ffhq_real_faces": 0,
        "AIS-4SD/StableDiffusion-3-faces-20250203-1545": 1,
        "SFHQ-T2I": 1
    }


clip_processor = CLIPProcessor.from_pretrained(Config.pretrained_model_name)
clip_mean = clip_processor.image_processor.image_mean
clip_std = clip_processor.image_processor.image_std


def get_all_samples(data_root_dir: Path) -> list[tuple[Path, int]]:
    """Get a list of samples, where each sample consists of the path to the image and the label"""
    samples = []
    for class_label_int in list(LABELS_DICT.values()):
        folder_paths = [folder_path for folder_path, class_label in IMAGE_FOLDER_NAMES_TO_LABELS.items() if class_label==class_label_int]
        for folder_path in folder_paths:
            for image_name in os.listdir(data_root_dir / folder_path):
                if image_name.lower().endswith((".png", ".jpg")):
                    image_path = data_root_dir / folder_path / image_name
                    samples.append((image_path, class_label_int))
    return samples


def get_train_transforms(config: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size=config.target_image_size),
        transforms.RandomApply([
            transforms.RandomHorizontalFlip()
        ], p=config.prob_horizontal_flip),
        transforms.RandomApply([
            transforms.v2.RandomResizedCrop(size=config.target_image_size)
        ], p=config.prob_random_crop),
        transforms.RandomApply([
            transforms.v2.JPEG(config.jpeg_quality_range)
        ], p=config.prob_jpeg_compress),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=config.gaussian_blur_kernel_size)
        ], p=config.prob_blur),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=clip_mean,
            std=clip_std
        )
    ])


def get_val_test_transforms(config: Config) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size=config.target_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=clip_mean,
            std=clip_std
        )
    ])


class FaceImageDataset:

    def __init__(self, samples: list[tuple[Path, int]], transform: transforms.Compose):
        """
        Args:
            samples: list of samples where each sample consists of the path to the image and the label
            transforms: the PyTorch transforms to apply to the images in the sample
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        """Return the total number of samples"""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """
        Get one sample.
        Returns:
            transformed_image_tensor: Image tensor output by the transform function
            label: 0 for real, 1 for synthetic
        """
        image_path, label = self.samples[idx]
        image = Image.open(image_path)
        transformed_image_tensor = self.transform(image)
        assert transformed_image_tensor.shape[0] == 3, "Unexpected number of channels; expected 3 for RGB."
        return transformed_image_tensor, label 


def load_sample_subsets(all_samples: list[tuple[Path, int]], config: Config) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    # shuffle samples
    train_size = int(config.train_split * len(all_samples))
    val_size = (len(all_samples) - train_size) // 2

    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]
    return train_samples, val_samples, test_samples


def load_train_dataset(train_samples: list, train_transforms: transforms.Compose) -> FaceImageDataset:
    train_dataset = FaceImageDataset(train_samples, train_transforms)
    logger.info(f"Loaded train_dataset with {len(train_dataset)} samples.")
    return train_dataset


def load_val_dataset(val_samples: list, val_transforms: transforms.Compose) -> FaceImageDataset:
    val_dataset = FaceImageDataset(val_samples, val_transforms)
    logger.info(f"Loaded val_dataset with {len(val_dataset)} samples.")
    return val_dataset


def load_test_dataset(test_samples: list, test_transforms: transforms.Compose) -> FaceImageDataset:
    test_dataset = FaceImageDataset(test_samples, test_transforms)
    logger.info(f"Loaded test_dataset with {len(test_dataset)} samples.")
    return test_dataset


def load_dataloaders(
    config: Config
    ) -> tuple[DataLoader, DataLoader, DataLoader]:

    all_samples = get_all_samples(config.data_root_dir)
    train_samples, val_samples, test_samples = load_sample_subsets(all_samples, Config)
    train_transforms = get_train_transforms(Config)
    val_transforms = get_val_test_transforms(Config)
    test_transforms = val_transforms

    train_dataset = load_train_dataset(train_samples, train_transforms)
    val_dataset = load_val_dataset(val_samples, val_transforms)
    test_dataset = load_test_dataset(test_samples, test_transforms)

    # add num workers?  
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader