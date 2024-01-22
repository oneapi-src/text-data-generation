# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Utility functions for running the transformers scripts.
"""

from datasets import load_dataset


def data_load(data_path, tokenizer, n_samples: int = 5000, val_prop: float = 0.2):
    """Load dataset of news headlines data.  Split into 80:20 train test.

    Args:
        data_path (str): relative path to csv file containing data
        tokenizer : tokenizer object to perform tokenization of the data
        n_samples (int) : number of total samples to use
        val_prop (float) : proportion of dataset to use for validation

    Returns:
        train, valid datasets for use in training
    """

    def tokenize_function(examples):
        return tokenizer("|HEADLINE| " + examples['headline_text'].rstrip() + ".", truncation=True)

    # Load the training set and tokenize each entry
    raw_dataset = load_dataset(
        "csv",
        data_files=data_path,
        split=f"train[:{n_samples}]"
    )

    # Create train and validation sets
    ds = raw_dataset.train_test_split(
        test_size=val_prop,
        seed=42
    )

    # Tokenize and add periods to each entry
    ds = ds.map(
        tokenize_function,
        remove_columns=['publish_date', 'headline_text'],
        load_from_cache_file=False
    )

    return ds['train'], ds['test']
