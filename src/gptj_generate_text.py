# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import logging
import time

'''
Perform text-generation using a GPTJ (LLM) model on Intel Sapphire Rapid Instances
'''

class GPTJModelF32:
    """
    Utilizes a FP32 model to generate text

    Args:
        model (str): Any valid huggingface model path
    """

    def __init__(self, model_name = "EleutherAI/gpt-j-6B"):
        self.model_name = model_name
        self.logger = logging.getLogger(self.model_name)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.load_model()

    # Creates a model instance for the object
    def load_model(self):
        
        self.logger.info('Loading Pretrained Model')
        self.logger.info("The first time the model is downloaded automatically and "
                        "it will take a considerable amount of time for the pretrained model to load.")
        self.logger.info('After the first run the model loading time is reduced.')
        tic = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        toc = time.time() - tic
        self.logger.info('Model Loaded: {}s'.format(toc))
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # Generates text completion for a provided prompt, along with the time taken
    # Returns - <prompt>,<time taken>
    def generate_text(self, prompt, max_new_tokens=32, temperature=0.9):
        prompt = [prompt] 

        self.logger.info('Generating Text')
        tic = time.time()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(input_ids, do_sample=True, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=self.tokenizer.eos_token_id)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        toc = time.time() - tic
        self.logger.info('Finished Generating Text: {}s'.format(toc))

        return gen_text, toc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', 
                        '-p', 
                        help="Prompt to be provided",
                        required=True, 
                        type=str)
    parser.add_argument('--max_new_tokens', 
                        help="Maximum no. of new tokens to be generated. Default - 32",
                        required=False, 
                        default=32,
                        type=int)
    parser.add_argument('--temperature', 
                        help="Temperature parameter for the GPT model. Default - 0.9",
                        required=False, 
                        default=0.9,
                        type=str)
    FLAGS = parser.parse_args()

    if not FLAGS.prompt:
        print("[Missing] Provide a prompt for the model")
    
    gptj = GPTJModelF32()
    res, _ = gptj.generate_text(FLAGS.prompt, max_new_tokens=FLAGS.max_new_tokens, temperature=FLAGS.temperature)
    print(res)
