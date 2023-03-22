import transformers
import os
import subprocess

module_path = transformers.__path__[0]
# get path for training_args.py
target_file_path = os.path.join(module_path, "training_args.py")

# apply patch to training_args.py file in the transformers package
subprocess.run(["patch", target_file_path, "transformers_xpu.patch"])

print("patch applied successfully")
