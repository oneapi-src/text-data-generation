# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
"""
Fine tune a selected Causal Language Model using the provided news headline
dataset
"""

import argparse
import time
import yaml

from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler
)
import torch

from utils import data_load


def main(flags):

    # Load pretrained tokenizer and model
    with open(flags.model_config, 'r', encoding='utf-8') as stream:
        conf = yaml.safe_load(stream)

    tokenizer = AutoTokenizer.from_pretrained(
        conf['model']['pretrained_model'])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        conf['model']['pretrained_model'], torchscript=True
    )

    # Reading in the data and create training arguments
    train, val = data_load(flags.data_path, tokenizer)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=flags.lr)

    # use IPEX to optimize model and optimizer for training
    if flags.intel:
        import intel_extension_for_pytorch as ipex
        if flags.bfloat16:
            model, optimizer = ipex.optimize(
                model, optimizer=optimizer, dtype=torch.bfloat16)
        else:
            model, optimizer = ipex.optimize(
                model, optimizer=optimizer, dtype=torch.float32)

    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=10,
        num_training_steps=flags.num_epochs * len(train)
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=flags.save_path,
        overwrite_output_dir=True,
        num_train_epochs=flags.num_epochs,
        evaluation_strategy="no",
        save_strategy="no",
        warmup_steps=10,
    )

    # Train the model with our dataset for Causal Language Modeling
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler)
    )

    start = time.time()
    trainer.train()
    end = time.time()
    trainer.save_model()

    print(f"Training time: {end - start}s")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_config',
                        type=str,
                        required=True,
                        help='yaml configuration file for model.')

    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='path to the "ABC million news headlines" csv.')

    parser.add_argument('--save_path',
                        type=str,
                        required=True,
                        help='path to save the model.')

    parser.add_argument('--num_epochs',
                        type=int,
                        required=False,
                        default=3,
                        help='number of epochs to train the model. Defaults to 3.')

    parser.add_argument('--lr',
                        type=int,
                        required=False,
                        default=5e-5,
                        help='learning rate for training. Defaults to 5e-5.')

    parser.add_argument('--intel',
                        required=False,
                        help="use intel pytorch extension to optimize model. Defaults to False.",
                        action="store_true",
                        default=False)

    parser.add_argument('--bfloat16',
                        required=False,
                        default=False,
                        action="store_true",
                        help="use bfloat16 for training. Defaults to False.")

    FLAGS = parser.parse_args()
    main(FLAGS)
