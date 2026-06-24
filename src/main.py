from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from dataset import PatchFeatureDataset
from training import Trainer


def _seed_everything(seed: int = 0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config: DictConfig):
    _seed_everything(config.experiment.seed)
    if not os.path.exists(config.experiment.directory):
        os.makedirs(config.experiment.directory, exist_ok=True)

    # Check for an existing checkpoint to optionally resume training from.
    checkpoint_path = os.path.join(config.experiment.directory, "model_best_loss.pt")
    last_checkpoint_path = os.path.join(config.experiment.directory, "model_last.pt")

    checkpoint_exists = os.path.isfile(checkpoint_path) or os.path.isfile(
        last_checkpoint_path
    )
    if config.experiment.resume and not checkpoint_exists:
        print("Warning: no checkpoint found. Training from scratch.")
    elif config.experiment.resume and checkpoint_exists:
        print("Resuming training from checkpoint.")
    elif not config.experiment.resume and checkpoint_exists:
        print(
            "Warning: a checkpoint exists but `experiment.resume` is not set."
            " Training from scratch."
        )

    # Prepare the dataset using the predefined dataset splits.
    dataset_labels = pd.read_csv(config.data.dataset_labels, index_col="slide_id")
    dataset_splits = pd.read_csv(config.data.dataset_splits)
    train_split = dataset_splits["train"].dropna().tolist()
    valid_split = dataset_splits["val"].dropna().tolist()
    test_split = dataset_splits["test"].dropna().tolist()

    # Collect the files and add them to the corresponding dataset group. Both h5py and
    # pickle files are supported (precomputed features and raw JPEG bytes, respectively).
    train_filenames, valid_filenames, test_filenames = [], [], []
    train_labels, valid_labels, test_labels = [], [], []
    for filename in glob.glob(os.path.join(config.data.directory, "*")):
        if not filename.endswith(".h5") and not filename.endswith(".pkl"):
            continue

        slide_name = ".".join(os.path.basename(filename).split(".")[:-1])

        # Assign the file to its split.
        if slide_name in train_split:
            train_filenames.append(filename)
            train_labels.append(dataset_labels.loc[slide_name, "type"])
        if slide_name in valid_split:
            valid_filenames.append(filename)
            valid_labels.append(dataset_labels.loc[slide_name, "type"])
        if slide_name in test_split:
            test_filenames.append(filename)
            test_labels.append(dataset_labels.loc[slide_name, "type"])

    train_dataset = PatchFeatureDataset(
        train_filenames,
        train_labels,
        column=config.data.dataset_column,
        max_length=config.data.max_length,
    )
    valid_dataset = PatchFeatureDataset(
        valid_filenames,
        valid_labels,
        column=config.data.dataset_column,
        max_length=config.data.max_length,
    )
    test_dataset = PatchFeatureDataset(
        test_filenames,
        test_labels,
        column=config.data.dataset_column,
        max_length=config.data.max_length,
    )

    trainer = Trainer(config, (train_dataset, valid_dataset, test_dataset))
    trainer.train()
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    config.experiment.directory = os.path.join(
        config.experiment.directory,
        os.path.basename(args.config).split(".")[0],
        f"seed{config.experiment.seed}",
    )
    config.wandb.name = config.wandb.name.format(seed=config.experiment.seed)
    main(config)
