#%%
from ming.configuration_bailingmm import *
from pathlib import Path
path_ming = Path(__file__).parent.parent/"ming"

from transformers import AutoProcessor
from ming.modeling_bailingmm import BailingMMNativeForConditionalGeneration
import torch
model_path = "inclusionAI/Ming-Lite-Omni"
processor = AutoProcessor.from_pretrained(path_ming, trust_remote_code=True)
model = BailingMMNativeForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    load_image_gen=True,
    cache_dir=""
).to("cuda")
# %%
