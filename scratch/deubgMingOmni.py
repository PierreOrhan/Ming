#%%
from pathlib import Path
from ming.modeling_bailingmm import BailingMMNativeForConditionalGeneration
from ming.configuration_bailingmm import BailingMMConfig
import torch
model_path = "mingomni_onlylast/mingomni_1.5"
local_model_path = Path("/auto/data5/speechExposureEphys/pretrainedModels/")

config = BailingMMConfig.from_pretrained(local_model_path/model_path/"debugmodel")
#%%
model = BailingMMNativeForConditionalGeneration(config)
#%%
# model = BailingMMNativeForConditionalGeneration(local_model_path/model_path/"debugmodel",
#                     load_vlm=True,
#                     attn_implementation="flash_attention_2",
#                     load_image_gen=False,
#                     torch_dtype=torch.bfloat16,
#                     device_map = "auto",
#                     max_memory = {i:"10GiB" for i in range(torch.cuda.device_count())})
#%%
import ming
from ming.processing_bailingmm import BailingMMProcessor
processor = BailingMMProcessor.from_pretrained(ming.__path__[0], trust_remote_code=True)

from ANN.models.mingomni.postAnalyses.forActivity import DataCollatorForMingPretraining
collator = DataCollatorForMingPretraining(processor)
#%%
texts= ["Read and transcribe all visible text in the provided image exactly as it appears."]
from probe.analysers.VisualText.symbol_gen import VisualText
text_dict =  {"text":"hello world"}
image = VisualText(**text_dict).PIL().convert("RGB")
proc_images = [image]
proc_audios = None
messages = [
    {
        "role": "HUMAN",
        "content": [
        ],
    } for _ in range(max(len(texts), len(proc_images or []), len(proc_audios or [])))
]
# Fill messages with text, image, and audio data
for modality,input in zip(["text", "image", "audio"], [texts,proc_images, proc_audios]):
        if input is not None:
            for i, m in enumerate(messages):
                if input[i] is not None:
                    m["content"].append({"type":modality,modality: input[i]})
# 1. Format inputs using chat template
text = [
    collator.processor.apply_chat_template([c], add_generation_prompt=True)
    for c in messages
]# 2. Extract vision/audio data
image_inputs, video_inputs, audio_inputs = collator.processor.process_vision_info(messages)# list(map(list,zip(*[collator.processor.process_vision_info([c]) for c in messages])))
# #
    # Use processor (it expands multimodal special tokens internally)
inputs = collator.processor(
        images=image_inputs,
        videos=video_inputs,  # No video support in this collator
        audios=audio_inputs,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False
    )

#%%
inputs.pop("pixel_values_reference")
inputs.pop("image_gen_width")
inputs.pop("image_gen_height")
from transformers import GenerationConfig
inputs = inputs.to(model.device)
with torch.inference_mode():
    generation_config = GenerationConfig.from_dict({'no_repeat_ngram_size': 10,"use_cache":False})
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=False,
        eos_token_id=processor.gen_terminator,
        generation_config=generation_config,
        image_gen=False
    )
# %%
