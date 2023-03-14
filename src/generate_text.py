# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

"""
Script to generate text from a pre-trained model given a file of prompts.
"""

import argparse
import os
import json
import time
from typing import List, Tuple, Union
import yaml

import numpy as np
import onnxruntime as ort
import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    set_seed
)


def generate_text(
    tokenizer: PreTrainedTokenizer,
    model: Union[torch.nn.Module, ort.InferenceSession],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    min_length: int = 0,
    max_length: int = 10,
    algorithm: str = 'greedy',
    stop_token: str = '.') -> Tuple[List[int],float]:
    """Generate text using the provided model and algorithm.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer that corresponds to the
            provided pre-trained model type
        model (Union[torch.nn.Module, ort.InferenceSession]): the model to use
            to generate logits for the next token.  Can be either PyTorch or 
            ONNX formatted.
        input_ids (torch.Tensor): a sequence of starting input_ids to complete.
        attention_mask (torch.Tensor): a sequence of attention_masks to complete.
        min_length (int, optional): the minimum length of the sequence generated.
            Defaults to 0.
        max_length (int, optional): the maximum length of the sequence generated.
            Defaults to 10.
        algorithm (str, optional): The algorithm to use for generation. Available
            algorithms are ['greedy','sample']. Defaults to 'greedy'.
        stop_token (str, optional): The token to consider a stop token. Will be used
            to determine when to stop and return. Defaults to '.'.

    Returns:
        Tuple(List[int], float): generated sequence of token_ids, total time for
            of model inference calls
    """
    

    all_token_ids = input_ids.clone()
    all_attention_masks = attention_mask.clone()
    eos_token_id = tokenizer([stop_token], return_tensors='np')[
        'input_ids'][0][0]
    has_eos = torch.zeros(1, dtype=torch.bool)

    total_time = 0
    for step in range(max_length):

        if isinstance(model, torch.nn.Module):
            start = time.time()
            next_token_logits = torch.nn.functional.softmax(
                model(
                    input_ids=all_token_ids,
                    attention_mask=all_attention_masks)[:, -1, :], dim=1)
            end = time.time()
            total_time += end - start
        elif isinstance(model, ort.InferenceSession):
            ort_input = {
                "input_ids": np.array(all_token_ids),
                "attention_mask": np.array(all_attention_masks)
            }
            start = time.time()
            next_token_logits = torch.nn.functional.softmax(
                torch.from_numpy(model.run(None, ort_input)[0])[:, -1, :], dim=1)
            end = time.time()
            total_time += end - start
        if algorithm == 'sample':
            next_tokens = torch.multinomial(next_token_logits, 1)[0]
            while step < min_length and next_tokens == eos_token_id:
                next_tokens = torch.multinomial(next_token_logits, 1)[0]
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat(
            [all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        all_attention_masks = torch.cat(
            [all_attention_masks, torch.ones([1, 1])], dim=-1).type_as(all_attention_masks)
        if step > min_length and next_tokens == eos_token_id:
            break

    return all_token_ids, total_time


def generate(
        tokenizer: PreTrainedTokenizer,
        model: Union[torch.nn.Module, ort.InferenceSession],
        min_length: int = 0,
        max_length: int = 10,
        prompt_file: str = None,
        benchmark_mode: bool = False,
        n_runs: int = 100):

    # read prompts from file into a list for batch processing
    tokenized_input = []
    if not benchmark_mode and prompt_file is not None:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            for prompt in f.readlines():
                text = prompt.rstrip()
                texts = ["|HEADLINE| " + text]
                tokenized_input.append(tokenizer(texts, return_tensors="pt"))
    else:
        tokenized_input.append(tokenizer(["|HEADLINE| "], return_tensors="pt"))

    # generate text from every tokenized prompt using the model
    tokenized_output = []
    if benchmark_mode:
        # run benchmarks on fixed input
        times = []
        set_seed(42)
        for i in range(n_runs):
            for tokenized_prompt in tokenized_input:
                _, total_time = generate_text(
                    tokenizer,
                    model,
                    tokenized_prompt.input_ids,
                    tokenized_prompt.attention_mask,
                    min_length=max_length,
                    max_length=max_length,
                    algorithm='greedy')
                if i > 10:
                    times.append(total_time)
        print(f"Average Generation time: {np.mean(times)}s")
    else:
        # the "." is assumed to represent the EOS
        # prevent it from generating eos too early and forces it to
        # be the last token generated to enforce min-length/max-length
        tokenized_output = []
        for tokenized_prompt in tokenized_input:
            res, _ = generate_text(
                tokenizer,
                model,
                tokenized_prompt.input_ids,
                tokenized_prompt.attention_mask,
                min_length=min_length,
                max_length=max_length,
                algorithm='sample')
            tokenized_output.append(res)

        out_json = []
        for output in tokenized_output:
            out = {}
            generation = tokenizer.batch_decode(
                output, skip_special_tokens=True)
            out["sentences"] = [g.split("|HEADLINE| ")[1] for g in generation]
            out_json.append(out)

        # Decode the tokenized output to text
        print(json.dumps(out_json))


def main(flags):

    # parse the yaml model config
    with open(flags.model_config, 'r', encoding='utf-8') as stream:
        conf = yaml.safe_load(stream)

    if not flags.benchmark_mode:
        min_length = conf['generation'].get('min_length', 10)
        max_length = min_length + \
            conf['generation'].get('max_length_buffer', 10)
    else:
        min_length = flags.benchmark_seq_length
        max_length = flags.benchmark_seq_length
    prompt_file = conf['generation'].get('prompt_file', None)

    if conf['model']['format'] == 'onnx':
        if 'path' in conf['model'] and conf['model']['path'] is not None:
            model = ort.InferenceSession(
                os.path.join(conf['model']['path'], "model.onnx"))
        else:
            print("No ONNX path provided. "
                  "The pretrained model must first be converted to "
                  "ONNX format and the saved path provided under "
                  "the 'onnx_path' in the configuration file to use.")
            return
    elif conf['model']['format'] == 'default':
        model = AutoModelForCausalLM.from_pretrained(
            conf['model']['path'])
    else:
        print("Model format not supported.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        conf['model']['pretrained_model'])
    tokenizer.pad_token = tokenizer.eos_token
    generate(
        tokenizer=tokenizer,
        model=model,
        min_length=min_length,
        max_length=max_length,
        prompt_file=prompt_file,
        benchmark_mode=flags.benchmark_mode
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_config',
                        required=True,
                        help="yaml configuration file for model.",
                        type=str)

    parser.add_argument('--benchmark_mode',
                        required=False,
                        help="use intel pytorch extension to optimize model.",
                        action="store_true",
                        default=False)

    parser.add_argument('--benchmark_seq_length',
                        required=False,
                        help="length of generation if benchmark mode is used.",
                        type=int,
                        default=10
                        )

    FLAGS = parser.parse_args()
    main(FLAGS)
