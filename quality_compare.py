"""
Quality comparison: pt (base) vs mix (instruction-tuned) PaliGemma.

Shows that output QUALITY is driven by the CHECKPOINT (pt vs mix), not by our
optimizations or parameter count. Both are 3B, same architecture, same code —
only the trained weights differ.

Usage:
    python quality_compare.py \
        --pt_path   "$HOME/projects/paligemma-weights/paligemma-3b-pt-224" \
        --mix_path  "$HOME/projects/paligemma-weights/paligemma-3b-mix-224" \
        --image_file_path test_images/pic1.jpeg
"""

import torch
import fire
from PIL import Image

from processing_paligemma import PaliGemmaProcessor
from modeling_gemma import KVCache, set_use_sdpa
from utils import load_hf_model

PROMPTS = [
    "caption en",
    "answer en what animal is this?",
    "answer en what color is the dog?",
    "answer en where is the dog?",
    "describe the image",                 # free-form (base will struggle, mix better)
]


@torch.no_grad()
def _gen(model, processor, device, prompt, image, max_new=40):
    ins = processor(text=[prompt], images=[image])
    ins = {k: v.to(device) for k, v in ins.items()}
    iid, am, pv = ins["input_ids"], ins["attention_mask"], ins["pixel_values"]
    kv = KVCache(); stop = processor.tokenizer.eos_token_id; toks = []
    for _ in range(max_new):
        o = model(input_ids=iid, pixel_values=pv, attention_mask=am, kv_cache=kv)
        kv = o["kv_cache"]
        nt = torch.argmax(o["logits"][:, -1, :], dim=-1, keepdim=True)
        if nt.item() == stop:
            break
        toks.append(nt.item()); iid = nt
        am = torch.cat([am, torch.ones((1, 1), device=device)], dim=-1)
    return processor.tokenizer.decode(toks, skip_special_tokens=True)


def _load(path, device):
    set_use_sdpa(True)
    model, tok = load_hf_model(path, device)
    model = model.to(device).to(torch.bfloat16).eval()
    proc = PaliGemmaProcessor(tok, model.config.vision_config.num_image_tokens,
                              model.config.vision_config.image_size)
    return model, proc


def main(pt_path=None, mix_path=None, image_file_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(image_file_path).convert("RGB")

    print("Loading pt (base) ..."); pt_model, pt_proc = _load(pt_path, device)
    print("Loading mix (instruction-tuned) ..."); mix_model, mix_proc = _load(mix_path, device)

    print("\n" + "=" * 78)
    print(f"{'PROMPT':<34} | {'pt (base)':<20} | mix (tuned)")
    print("-" * 78)
    for p in PROMPTS:
        pt_out = _gen(pt_model, pt_proc, device, p, image)
        mix_out = _gen(mix_model, mix_proc, device, p, image)
        print(f"{p[:33]:<34} | {pt_out[:20]:<20} | {mix_out}")
    print("=" * 78)
    print("\nSame 3B params, same architecture, same code — only the trained weights")
    print("differ. 'mix' (instruction-tuned) gives better free-form quality.")


if __name__ == "__main__":
    fire.Fire(main)
