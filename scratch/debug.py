#%%
from ming.configuration_bailingmm import *
from pathlib import Path
path_ming = Path("/lustre/fswork/projects/rech/dab/uzz43va")/"MING"/"ming"

from transformers import AutoProcessor
from ming.modeling_bailingmm import BailingMMNativeForConditionalGeneration
import torch
model_path = "inclusionAI/Ming-Lite-Omni-1.5"
local_model_path = "/lustre/fsn1/projects/rech/dab/uzz43va/NeuroData/pretrainedModels/ming15_lite_onlylast/Ming-Lite-Omni-1.5"
# processor = AutoProcessor.from_pretrained(path_ming, trust_remote_code=True)
model = BailingMMNativeForConditionalGeneration.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    load_image_gen=False,
    device_map = "auto",
    max_memory = {i:"10GiB" for i in range(torch.cuda.device_count())}
)
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
image_processor = processor.image_processor  # generic vision preprocessor
# %%
from typing import  Dict, List, Union
import torch
import torch.utils.checkpoint
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any
import torch
from transformers import AutoTokenizer, AutoProcessor
from ming.processing_bailingmm import BailingMMProcessor

@dataclass
class DataCollatorForMingPretraining:
    """
    Data collator using the BailingMM (Ming) processor.

    Input feature list element: (image, audio, text)
      image: PIL.Image.Image | torch.Tensor | None
      audio: torch.Tensor 1D waveform | None
      text : str | None  (None -> single pad token)

    Rules:
      - If any image in the batch is None -> no images passed (pixel_values=None).
      - If any audio in the batch is None -> no audios passed (audio fields absent).
      - Missing/empty text replaced by a single pad token.
      - Labels = input_ids with padding positions masked to -100.
    """
    processor: BailingMMProcessor
    audio_sample_rate: int = 16000  # adjust if different

    def __post_init__(self):
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.add_special_tokens({"pad_token": "<pad>"})

    def _prepare_texts(self, texts: List[Optional[str]]) -> List[str]:
        pad_tok = self.processor.tokenizer.pad_token or "<pad>"
        return [pad_tok if (t is None or t == "") else t for t in texts]

    def _package_audios(self, audios: List[torch.Tensor]) -> List[Tuple[torch.Tensor, int]]:
        # Processor audio interface expects list of (waveform, sr) or similar.
        packaged = []
        for w in audios:
            if not isinstance(w, torch.Tensor):
                w = torch.tensor(w)
            if w.dim() != 1:
                raise ValueError("Audio waveform must be 1D (T,).")
            packaged.append((w, self.audio_sample_rate))
        return packaged

    def __call__(
        self,
        features: List[Tuple[
            Optional[Union["PIL.Image.Image", torch.Tensor]],
            Optional[torch.Tensor],
            Optional[str]
        ]]
    ) -> Dict[str, torch.Tensor]:

        images = [f[0] for f in features]
        audios = [f[1] for f in features]
        texts  = [f[2] for f in features]

        # Texts
        texts = self._prepare_texts(texts)

        # Decide whether to include images
        if any(im is None for im in images):
            proc_images = None
        else:
            proc_images = images  # pass full list

        # Decide whether to include audios
        if any(a is None for a in audios):
            proc_audios = None
        else:
            proc_audios = self._package_audios(audios)

        # Use processor (it expands multimodal special tokens internally)
        batch_feat = self.processor(
            images=proc_images,
            videos=None,  # No video support in this collator
            audios=proc_audios,
            text=texts,
        )
        return batch_feat

        # input_ids = batch_feat["input_ids"]
        # attention_mask = batch_feat.get("attention_mask", torch.ones_like(input_ids))

        # # Labels: mask padding
        # labels = input_ids.clone()
        # if attention_mask is not None:
        #     labels[attention_mask == 0] = -100

        # output = {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "labels": labels,
        #     "use_cache": False,
        # }

        # # Forward optional multimodal tensors if present
        # for k in [
        #     "pixel_values",
        #     "image_grid_thw",
        #     "pixel_values_videos",
        #     "video_grid_thw",
        #     "encoder_feats",            # audio processor possible keys
        #     "audio_feats",              # depending on implementation
        #     "audio_placeholder_loc_lens",
        # ]:
        #     if k in batch_feat:
        #         output[k] = batch_feat[k]

        # return output

# Example setup (outside the class):
# processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
# tokenizer = processor.tokenizer
# mm_processor = processor  # Already a BailingMMProcessor instance via trust_remote_code
# collator = DataCollatorForMingPretraining(processor=mm_processor, tokenizer=tokenizer)
# dataloader = DataLoader(dataset, batch_size=..., collate_fn=collator)