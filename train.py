import logging
from fake_image_detection.config import Config
from fake_image_detection.dataset import (
    get_all_samples,
    load_sample_subsets,
    load_dataloaders
)

logger_config = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(logger_config)

train_dataloader, val_dataloader, test_dataloader = load_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
)