# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415,E0401,R0914

import transformers
import os
import subprocess

module_path = transformers.__path__[0]
# get path for training_args.py
target_file_path = os.path.join(module_path, "training_args.py")

# apply patch to training_args.py file in the transformers package
subprocess.run(["patch", target_file_path, "transformers_xpu.patch"])

print("patch applied successfully")
