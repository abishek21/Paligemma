# PaliGemma Inference — Course Notes

A complete, code-anchored walkthrough of what happens from the moment you pass an
**image + prompt** until the model prints text. Every step points to the exact
file, function, and line so you can read the code alongside these notes.

> Companion docs: `BENCHMARKING.md` (prefill/decode/KV-cache metrics) and
> `stage2_vllm/` (production serving). This file is the *architecture* deep-dive.

---

## 0. The 10,000-foot view

PaliGemma = **SigLIP vision encoder** + **linear projector** + **Gemma language
model (decoder-only transformer)**.

```
        image ───► SigLIP ViT ───► projector ─┐
                                               ├─► merge into one sequence ─► Gemma decoder ─► logits ─► token
  "caption en" ─► tokenizer ─► text embeds ───┘        (image tokens + text tokens)
```

The whole thing is an **autoregressive image-conditioned text generator**. The
image becomes a block of "soft tokens" that sit at the front of the sequence; the
text prompt follows; Gemma then predicts the next token over and over.

### The 4 source files
| File | Role |
|------|------|
| `inference.py` | CLI + the generation loop (prefill then decode). |
| `processing_paligemma.py` | Turns image+text into tensors (`pixel_values`, `input_ids`). |
| `modeling_siglip.py` | The vision encoder (ViT). |
| `modeling_gemma.py` | The projector, the Gemma LM, the KV cache, and the multimodal merge. |
| `utils.py` | Loads weights from safetensors into the model. |

---

## 1. Entry point — where your inputs land

**File: `inference.py`**

- `main()` (line ~113): picks the device, calls `load_hf_model` (`utils.py`) to
  build the model, then builds the `PaliGemmaProcessor` and calls `test_inference`.
- `get_model_inputs()` (line ~15): opens the image with PIL, wraps prompt/image in
  lists, and calls the processor:
  ```python
  model_inputs = processor(text=[prompt], images=[image])
  ```
- The processor returns a dict with **three tensors**:
  - `pixel_values`  → `[1, 3, 224, 224]` (the preprocessed image)
  - `input_ids`     → `[1, seq_len]` (image placeholder tokens + text tokens)
  - `attention_mask`→ `[1, seq_len]` (all ones here — no padding)

That's the boundary: after this point everything is tensors.

---

## 2. Preprocessing — turning pixels & text into tokens

**File: `processing_paligemma.py`**

### 2a. The image path — `process_images()` (line ~55)
Applied to the PIL image, in order:
1. **resize** to 224×224 (`resize`, line ~35)
2. **rescale** pixel values to [0,1] via `× 1/255` (`rescale`, line ~24)
3. **normalize** with mean/std = 0.5 → range ≈ [-1, 1] (`normalize`, line ~44)
4. **transpose** HWC → CHW so shape is `[3, 224, 224]`

Result is stacked to `[1, 3, 224, 224]` and turned into a torch tensor.

### 2b. The text path — `PaliGemmaProcessor.__call__()` (line ~112)
The crucial trick is **`add_image_tokens_to_prompt()`** (line ~10):
```python
return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"
```
So the string fed to the tokenizer is literally:
```
<image><image>…(256 times)…<image><bos>caption en\n
```
- `image_seq_len` = **256** = `(224/16)² = 14×14` patches (computed in
  `PaliGemmaConfig.__init__`, `modeling_gemma.py` line ~103).
- Those 256 `<image>` tokens are **placeholders** — real image features get
  scattered into them later (Section 5).
- Special tokens `<image>`, 1024 `<locNNNN>` (detection boxes) and 128 `<segNNN>`
  (segmentation) are registered in `__init__` (lines ~90-105).

**Output:** `input_ids` = `[256 image-placeholder ids] + [bos] + [text ids] + [\n id]`.

---

## 3. Vision encoder — SigLIP ViT

**File: `modeling_siglip.py`** — invoked from
`PaliGemmaForConditionalGeneration.forward` (`modeling_gemma.py` line ~540):
```python
selected_image_feature = self.vision_tower(pixel_values.to(...))
```

Call chain: `SiglipVisionModel` → `SiglipVisionTransformer` → embeddings → encoder → post-layernorm.

### 3a. Patch embeddings — `SiglipVisionEmbeddings.forward()` (line ~60)
- A **Conv2d** with `kernel=stride=patch_size(14)` chops the image into
  16×16 = **256 patches**, each projected to `hidden_size=1152`. This 256 matches
  the PaliGemma image-token count exactly (`num_image_tokens: 256`).
- Flatten + transpose → `[1, 256, 1152]`.
- **Add learned position embeddings** (`nn.Embedding`, line ~52). ViT positions
  are learned & absolute — *no causal masking* because an image has no "future".

### 3b. Transformer stack — `SiglipEncoder` (line ~196)
`num_hidden_layers` identical `SiglipEncoderLayer`s (line ~162). Each is
**pre-norm**:
```
x = x + Attention(LayerNorm(x))     # SiglipAttention, line ~77
x = x + MLP(LayerNorm(x))           # SiglipMLP (gelu-tanh), line ~144
```
- `SiglipAttention.forward()` (line ~94): standard multi-head self-attention,
  **full/bidirectional** (every patch attends to every patch). No KV cache here —
  the image is encoded once.

**Output:** `[1, num_patches, 1152]` contextualized image features.

---

## 4. The projector — matching dimensions

**File: `modeling_gemma.py`, `PaliGemmaMultiModalProjector` (line ~427)**
```python
self.linear = nn.Linear(vision_hidden(1152), projection_dim(2048))
```
One `nn.Linear` maps each image feature from the vision width (1152) to the
**Gemma hidden size (2048)** so image and text embeddings live in the same space.
Called at `modeling_gemma.py` line ~543:
```python
image_features = self.multi_modal_projector(selected_image_feature)
```

---

## 5. ⭐ The merge — where vision meets language (the heart of PaliGemma)

**File: `modeling_gemma.py`, `_merge_input_ids_with_image_features()` (line ~453)**

This is the single most important function for understanding multimodality.

1. **Text embeddings** first: back in `forward` (line ~536)
   ```python
   inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
   ```
   This embeds *all* ids — including the 256 `<image>` placeholders (which get
   garbage embeddings we're about to overwrite).

2. **Scale image features** (line ~460):
   ```python
   scaled_image_features = image_features / (hidden_size ** 0.5)
   ```

3. **Build masks** (lines ~466-472) to know which positions are text / image / pad:
   ```python
   text_mask  = (input_ids != image_token_index) & (input_ids != pad)
   image_mask = (input_ids == image_token_index)
   pad_mask   = (input_ids == pad)
   ```

4. **Assemble the final embedding** (lines ~475-481):
   ```python
   final = where(text_mask,  inputs_embeds, zeros)     # keep text embeds
   final = final.masked_scatter(image_mask, scaled_image_features)  # drop image feats into <image> slots
   final = where(pad_mask, zeros, final)               # zero padding
   ```
   `masked_scatter` is used (not `where`) because the image features are a
   *different sequence length* than the full sequence — it streams the 256 image
   vectors into the 256 placeholder positions in order.

**Result:** one unified `[1, seq_len, 2048]` embedding = `[image feats | text embeds]`.

### 5a. The attention mask & positions (same function, lines ~485-517)
This also builds the causal mask and `position_ids`, and it's **where the two
stages diverge**:

- **Prefill** (`kv_cache is None or empty`, line ~488): mask is all-zeros of shape
  `[batch, q_len, q_len]` — i.e. *no masking*. PaliGemma lets the prompt (image +
  text) attend **fully bidirectionally**; only the generated suffix is causal.
- **Decode** (line ~495): `q_len == 1`, mask shape `[batch, 1, kv_len]`, again all
  zeros because the single new token may attend to everything cached.
- `position_ids` (lines ~508-516): built from `attention_mask.cumsum(-1)`. In
  decode it's just the last position.

---

## 6. The Gemma language model — layers & decode

**File: `modeling_gemma.py`**

Call chain: `GemmaForCausalLM.forward` (line ~396) → `GemmaModel.forward`
(line ~353) → N × `GemmaDecoderLayer` (line ~291) → final `GemmaRMSNorm` → `lm_head`.

### 6a. `GemmaModel.forward()` (line ~353)
- Multiplies embeddings by `sqrt(hidden_size)` — Gemma's input normalizer
  (line ~366).
- Loops over `self.layers` (the decoder stack), passing `attention_mask`,
  `position_ids`, and the shared `kv_cache`.
- Applies final RMSNorm.

### 6b. One decoder layer — `GemmaDecoderLayer.forward()` (line ~303)
Pre-norm transformer block (same shape as SigLIP but **causal** and with RoPE):
```
x = x + SelfAttn(RMSNorm(x))     # GemmaAttention
x = x + MLP(RMSNorm(x))          # GemmaMLP (gated gelu)
```
- `input_layernorm` / `post_attention_layernorm` = **`GemmaRMSNorm`** (line ~108).
  Note Gemma's quirk: `output * (1 + weight)` (line ~120).
- `GemmaMLP` (line ~175) is a **gated** MLP: `down(gelu(gate(x)) * up(x))`.

### 6c. Attention — `GemmaAttention.forward()` (line ~230)  ← KV CACHE LIVES HERE
Step by step:
1. Project to Q, K, V (lines ~239-244). **Grouped-Query Attention**:
   `num_attention_heads` (Q) > `num_key_value_heads` (K/V), so K/V are smaller.
2. **RoPE** rotary position encoding applied to Q and K
   (`apply_rotary_pos_emb`, line ~168; embeddings from `GemmaRotaryEmbedding`,
   line ~124). This is how Gemma encodes position — rotating Q/K by an angle
   proportional to their position.
3. **KV cache update** (lines ~256-257):
   ```python
   if kv_cache is not None:
       key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)
   ```
4. **`repeat_kv`** (line ~193) expands the few KV heads to match the Q heads (GQA).
5. Scaled dot-product attention `softmax(QKᵀ/√d + mask) · V` (lines ~262-277).
6. Output projection `o_proj`.

### 6d. The KV cache itself — `KVCache` (line ~8)
```python
class KVCache:
    def update(self, key_states, value_states, layer_idx):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)          # prefill: create
        else:
            self.key_cache[layer_idx] = torch.cat(     # decode: append
                [self.key_cache[layer_idx], key_states], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
```
- Shape per layer: `[batch, num_kv_heads, seq_len, head_dim]`.
- `num_items()` (line ~14) returns the cached sequence length (`shape[-2]`).
- **Why it exists:** in decode we feed only 1 new token; without the cache we'd
  recompute K/V for the entire history every step (O(n²) → O(n)).

### 6e. The head — `GemmaForCausalLM` (line ~381)
- `lm_head` (line ~387): `Linear(hidden(2048) → vocab(257152))` producing logits.
- `tie_weights()` (line ~392): the `lm_head` **shares the embedding matrix** —
  wired up in `utils.load_hf_model`.

**Output:** `logits` `[1, seq_len, vocab]` (+ the updated `kv_cache`).

---

## 7. The generation loop — the two stages in action

**File: `inference.py`, `test_inference()` (line ~28)**

```python
kv_cache = KVCache()
for _ in range(max_tokens_to_generate):
    outputs = model(input_ids=..., pixel_values=..., attention_mask=..., kv_cache=kv_cache)
    kv_cache = outputs["kv_cache"]
    next_token_logits = outputs["logits"][:, -1, :]   # only the LAST position matters
    next_token = argmax(...) or _sample_top_p(...)     # greedy or nucleus sampling
    if next_token == eos: break
    input_ids = next_token                             # feed just the new token
    attention_mask = cat([attention_mask, ones(1,1)])  # grow the mask by 1
```

- **Iteration 1 = PREFILL.** `input_ids` is the full prompt (image+text),
  `kv_cache` is empty. One big forward pass fills the cache and yields the 1st token.
  Latency here = **Time To First Token**.
- **Iterations 2…N = DECODE.** `input_ids` is a single token; the cache supplies
  all history. Each pass yields one token. Latency here = **inter-token latency**.
- `pixel_values` is passed every step in this simple loop, but the image only
  actually matters at prefill (the image features already live in the KV cache
  afterwards). Production servers skip re-encoding — see `BENCHMARKING.md`.

`_sample_top_p` (line ~92) implements nucleus sampling; greedy `argmax` is the
default (`do_sample=False`).

---

## 8. Putting it all together — the full data flow

```
inference.py:main
  └─ utils.load_hf_model ................. build model, load safetensors, tie weights
  └─ processing_paligemma.PaliGemmaProcessor
        ├─ process_images ............... [1,3,224,224] pixel_values
        └─ add_image_tokens_to_prompt ... "<image>*256 <bos> prompt \n" → input_ids
  └─ test_inference (loop)
        └─ PaliGemmaForConditionalGeneration.forward
              ├─ language_model.embed_tokens(input_ids) ... text+placeholder embeds
              ├─ vision_tower(pixel_values) .............. SigLIP → image features
              │     └─ SiglipVisionEmbeddings → SiglipEncoder(×L) → post_layernorm
              ├─ multi_modal_projector .................... 1152 → 2048
              ├─ _merge_input_ids_with_image_features ..... scatter image feats into <image> slots
              │                                             + build mask + position_ids
              └─ language_model (GemmaForCausalLM)
                    └─ GemmaModel: ×N GemmaDecoderLayer
                          ├─ GemmaAttention (RoPE + GQA + KVCache.update)
                          └─ GemmaMLP (gated gelu)
                    └─ RMSNorm → lm_head → logits
        └─ pick next token → append → repeat
```

---

## 9. Where to look for each concept (cheat sheet)

| You want to understand… | Go to |
|--------------------------|-------|
| How image becomes tokens | `processing_paligemma.py` `process_images`, `add_image_tokens_to_prompt` |
| The vision transformer | `modeling_siglip.py` `SiglipVisionEmbeddings`, `SiglipEncoderLayer`, `SiglipAttention` |
| Vision→language bridge | `modeling_gemma.py` `PaliGemmaMultiModalProjector` (line 427) |
| **How image + text merge** | `_merge_input_ids_with_image_features` (line 453) |
| **Where the KV cache is** | `KVCache` (line 8) + used in `GemmaAttention.forward` (lines 256-257) |
| Positional encoding (text) | `GemmaRotaryEmbedding` (124), `apply_rotary_pos_emb` (168) |
| Grouped-Query Attention | `repeat_kv` (193) + Q/K/V projections in `GemmaAttention` (239) |
| The decoder layers | `GemmaDecoderLayer` (291), `GemmaModel` (335) |
| Normalization (Gemma) | `GemmaRMSNorm` (108) |
| Logits / vocabulary head | `GemmaForCausalLM.lm_head` (387), `tie_weights` (392) |
| **Prefill vs decode split** | mask branch in merge (lines 488-506) + loop in `inference.py` (line 46) |
| Sampling | `inference.py` `_sample_top_p` (92) |

---

## 10. Key numbers for the 3b-pt-224 model

| Thing | Value | Why |
|-------|-------|-----|
| Image resolution | 224×224 | the `-224` variant |
| Patch size | 14 | Conv2d kernel/stride |
| Image tokens | 256 = (224/14)² = 16×16 | placeholder `<image>` count |
| Vision hidden | 1152 | SigLIP width (27 layers) |
| LM hidden | 2048 | Gemma width (18 layers) |
| Attention heads (LM) | 8 Q / **1 KV** | extreme GQA (multi-query) → tiny KV cache |
| Vocab | 257,216 | includes 1024 loc + 128 seg tokens |
| Attention | Grouped-Query | fewer KV heads → smaller KV cache |
| Position enc. | RoPE (text), learned abs (vision) | — |
| Prompt attention | bidirectional | prefill mask is all-zeros |

---

### Suggested reading order for the course
1. `processing_paligemma.py` (inputs) →
2. `modeling_siglip.py` (vision) →
3. `_merge_input_ids_with_image_features` (the bridge) →
4. `GemmaAttention` + `KVCache` (the engine) →
5. `inference.py` loop (prefill/decode) →
6. `BENCHMARKING.md` (measure it) →
7. `stage2_vllm/` (scale it).
