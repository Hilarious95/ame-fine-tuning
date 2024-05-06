
import os
os.system("pip install unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git")
if version_flag == "new":
    os.system("pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes")
else:
    os.system("pip install --no-deps xformers trl peft accelerate bitsandbytes")

