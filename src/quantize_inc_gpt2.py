# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Quantize a model using intel extension for pytorch
"""

import argparse

import os
import onnxruntime as ort
import numpy as np
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
from neural_compressor.config import AccuracyCriterion, PostTrainingQuantConfig
from neural_compressor import quantization
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from tqdm import tqdm
import yaml

from utils import data_load


def evaluate(model, val_dataloader):

    # Eval!
    eval_loss = 0.0
    nb_eval_steps = 0

    options = ort.SessionOptions()
    session = ort.InferenceSession(
        model.SerializeToString(),
        options,
        providers=ort.get_available_providers())

    len_inputs = len(session.get_inputs())

    inputs_names = [session.get_inputs()[i].name for i in range(len_inputs)]
    ort_inputs = {}
    for _, batch in enumerate(tqdm(val_dataloader, desc="Evaluating")):

        for i in range(len_inputs):
            ort_inputs.update({
                inputs_names[i]: np.array(batch[inputs_names[i]])
            })
        labels = batch['labels']

        predictions = session.run(None, ort_inputs)
        lm_logits = predictions[0]
        lm_logits = torch.from_numpy(lm_logits)

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))

        eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    return result['perplexity'].item()


def quantize_model(model, tokenizer, val_data):
    """Quantizes the model using the given dataset and INC config

    Args:
        model : Pre-trained model to quantize.
        tokenizer : Pre-trained tokenizer to use for evaluation.
        val_data : Validation dataset for accuracy aware evaluation.
    """

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    # Load up a trainer for model evaluation Causal Language Modeling
    eval_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(
        val_data,
        sampler=eval_sampler,
        batch_size=10,
        collate_fn=data_collator
    )

    def evaluate_perplexity_loss(model_q) -> float:
        return evaluate(model_q, val_dataloader)

    accuracy_criterion = AccuracyCriterion()
    accuracy_criterion.higher_is_better = False
    accuracy_criterion.relative = 0.11
    config = PostTrainingQuantConfig(approach='dynamic',
                                     op_name_list={'MatMul_2924': {
                                         'activation': {'dtype': ['fp32']},
                                         'weight': {'dtype': ['fp32']}
                                     }},
                                     accuracy_criterion=accuracy_criterion)
    q_model = quantization.fit(model,
                               config,
                               eval_func=evaluate_perplexity_loss)
    return q_model


def main(flags) -> None:
    """Calibrate model for int 8 and serialize as a .pt

    Args:
        flags: benchmarking flags
    """

    # Load pretrained tokenizer and ONNX model
    # parse the yaml model config
    with open(flags.model_config, 'r', encoding='utf-8') as stream:
        conf = yaml.safe_load(stream)

    if conf['model']['format'] != 'onnx':
        print(
            f'Model needs to be in ONNX format.  Specified in config as {conf["model"]["format"]}.')
        return

    tokenizer = AutoTokenizer.from_pretrained(
        conf['model']['pretrained_model'])
    tokenizer.pad_token = tokenizer.eos_token

    # Load ONNX specifics based on the GPT2 model
    if 'path' in conf['model'] and conf['model']['path'] is not None:
        # GPT2 optimizer
        opt_options = BertOptimizationOptions('gpt2')
        opt_options.enable_embed_layer_norm = False
        model_optimizer = optimizer.optimize_model(
            os.path.join(conf['model']['path'], "model.onnx"),
            'gpt2',
            num_heads=12,
            hidden_size=768,
            optimization_options=opt_options)
    else:
        print("No ONNX path provided. "
              "For proper quantization, the saved model needs to be provided "
              "in ONNX format and the saved path provided under "
              "the 'path' in the configuration file.")

        return

    model = model_optimizer.model

    # Reading in the data and create training arguments
    _, val_data = data_load(flags.data_path, tokenizer)

    # Quantize model using Accuracy Aware Quantization
    quantized_model = quantize_model(
        model,
        tokenizer,
        val_data
    )
    if not os.path.exists(flags.save_model_dir):
        os.makedirs(flags.save_model_dir)

    quantized_model.save(os.path.join(flags.save_model_dir, "model.onnx"))
    tokenizer.save_pretrained(flags.save_model_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_config',
                        required=True,
                        help="yaml configuration file for model.",
                        type=str)

    parser.add_argument('--save_model_dir',
                        type=str,
                        required=True,
                        help="directory to save the quantized model to.")

    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='path to the "ABC million news headlines" csv.')

    FLAGS = parser.parse_args()

    main(FLAGS)
