
import os
os.system("pip3 install unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git")
if version_flag == "new":
    os.system("pip3 install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes")
else:
    os.system("pip3 install --no-deps xformers trl peft accelerate bitsandbytes")

