# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause


import argparse
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
        self.load_model()

    # Creates a model instance for the object
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # Generates text completion for a provided prompt, along with the time taken
    # Returns - <prompt>,<time taken>
    def generate_text(self, prompt, max_new_tokens=32, temperature=0.9, headline=False):
        prompt = [prompt] 
        tic = time.time()
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        toc = time.time() - tic
        print(gen_text, flush=True)
        return gen_text, toc



class GPTJModelI8:
    """
    Utilizes an INT8 IR model to generate text

    Args:
        ir_path (str): A valid path where an IR model resides
    """

    def __init__(self, ir_path="../int8_ir/"):
        self.ir_path = ir_path
        self.load_model()
        self.compile_ir_model()

    # Creates a model instance for the object
    def load_model(self):
        self.model_id = "EleutherAI/gpt-j-6B"
        config = AutoConfig.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(config)
        # Setting custom attributes
        setattr(self.model, "generate",  types.MethodType(itrex_generation_utils.GenerationMixin.generate, self.model))
        setattr(self.model, "beam_search", types.MethodType(itrex_generation_utils.GenerationMixin.beam_search, self.model))
        setattr(self.model, "_update_model_kwargs_for_generation",  types.MethodType(itrex_generation_utils.GenerationMixin._update_model_kwargs_for_generation, self.model))
        setattr(self.model, "_get_stopping_criteria", types.MethodType(itrex_generation_utils.GenerationMixin._get_stopping_criteria, self.model))
        setattr(self.model, "_extract_past_from_model_output", types.MethodType(itrex_generation_utils.GenerationMixin._extract_past_from_model_output, self.model))
        self.model.eval()

    # Uses the IR model
    def compile_ir_model(self):
        self.graph = compile(self.ir_path)
        print("Using IR file {}".format(self.ir_path))

    # Generates text completion for a provided prompt, along with the time taken
    # Returns - <prompt>,<time taken>
    def generate_text(self, prompt, max_new_tokens=32, temperature=0.9):
        prompt = [prompt]
        generate_kwargs = dict(do_sample=False, temperature=temperature, num_beams=4, past_kv_nums=28, llama=False)
        with torch.no_grad(), torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16):
            tic = time.time()
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            gen_tokens = self.model.generate(input_ids, max_new_tokens=max_new_tokens, engine_model = self.graph, **generate_kwargs) # provides the IR model as engine
            gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
            toc = time.time() - tic
            print(gen_text, flush=True)
        return gen_text, toc




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        '-m', 
                        help="Model to be run (int8|fp32)",
                        required=True, 
                        type=str)
    parser.add_argument('--prompt', 
                        '-p', 
                        help="Prompt to be provided",
                        required=True, 
                        type=str)
    parser.add_argument('--model_path', 
                        help="Path to the generated INT8 IR model. Required if model used is int8",
                        required=False, 
                        type=str)
    parser.add_argument('--max_new_tokens', 
                        help="Maximum no. of new tokens to be generated. Default - 32",
                        required=False, 
                        default=32,
                        type=int)
    parser.add_argument('--temperature', 
                        help="Teperature parameter for the GPT model. Default - 0.9",
                        required=False, 
                        default=0.9,
                        type=str)
    FLAGS = parser.parse_args()

    if not FLAGS.prompt:
        print("[Missing] Provide a prompt for the model")
    
    # For INT8 Quantized model execution
    if FLAGS.model == 'int8':
        if not FLAGS.model_path:
            print("[Missing] Provide a relative path to the IR model")
        else:
            import torch
            import types
            from accelerate import init_empty_weights
            import generation_utils as itrex_generation_utils
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
            from intel_extension_for_transformers.backends.neural_engine.compile import compile
            gptj = GPTJModelI8(FLAGS.model_path)
            res, sec = gptj.generate_text(FLAGS.prompt, max_new_tokens=FLAGS.max_new_tokens, temperature=FLAGS.temperature)
            print(res, sec)
    # For FP32 model execution
    elif FLAGS.model == 'fp32':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        gptj = GPTJModelF32()
        res, sec = gptj.generate_text(FLAGS.prompt, max_new_tokens=FLAGS.max_new_tokens, temperature=FLAGS.temperature)
        print(res, sec)
    else:
        print("[Invalid] Model provided is incorrect. Provide either 'int8' / 'fp32' ")
