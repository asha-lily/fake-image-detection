import logging
from fake_image_detection.config import Config
from fake_image_detection.dataset import (
    get_all_samples,
    load_sample_subsets,
    load_dataloaders
)

log_config = logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(log_config)


### get data_root_dir from config?
all_samples = get_samples(data_root_dir)

### get train, val & test samples
train_samples, val_samples, test_samples = load_sample_subsets(all_samples, Config)

### get train, val test transforms
train_transforms = get_train_transforms(Config)
val_transforms = get_val_test_transforms(Config)
test_transforms = val_transforms

### get train, val & test datasets
train_dataset = load_train_dataset(train_samples, train_transforms)
val_dataset = load_val_dataset(val_samples, val_transforms)
test_dataset = load_test_dataset(test_samples, test_transforms)

### get train, val & test dataloaders
train_dataloader, val_dataloader, test_dataloader = load_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset
)