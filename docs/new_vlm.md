# Integrating a new VLM

FOCUS separates frame selection from generation. Choose the integration depth
that your model supports:

| Integration | Required access | What works |
|---|---|---|
| Gating only | Patch encoder plus any frame-to-text generator | Fewer VLM calls |
| Gating and feedback | Above, plus decoder attention and mutable value cache | Full closed loop |

A hosted model that only returns text cannot provide attention feedback or
value-cache calibration. It can still consume selected bursts, with feedback
explicitly disabled. The built-in factories recognize Qwen2-VL and Qwen3-VL;
a new Hugging Face model ID is not sufficient to add another architecture.

## 1. Supply patch embeddings

Subclass `BaseEncoder`, or implement `encode_frame` on your VLM adapter. Frames
arrive as OpenCV **BGR**, `uint8` arrays with shape `(height, width, 3)`.
Convert to RGB if required by your processor.

```python
from focus.encoders.base import BaseEncoder

class MyEncoder(BaseEncoder):
    def __init__(self, model, processor, device="cuda"):
        self.model = model
        self.processor = processor
        self.device = device
        self.hidden_size = model.config.hidden_size

    def __call__(self, frame):
        # Implement the model-specific preprocessing/vision forward pass.
        embeddings, grid_h, grid_w = self.extract_patches(frame)
        self.last_grid_hw = (grid_h, grid_w)
        return embeddings  # torch.Tensor, shape (grid_h * grid_w, hidden_size)
```

`extract_patches` above is a model-specific placeholder, not a library method.
Return corresponding spatial patches in a stable order and resolution. Exclude
CLS/register tokens, preserve post-merge grid information, and use inference
mode. A pooled whole-image vector does not supply spatial feedback. Verify
`len(embeddings) == grid_h * grid_w` before integration. Do not reuse a threshold
across encoders without checking their dissimilarity distributions.

## 2. Implement generation

Subclass [BaseVLMPipeline](../src/focus/vlm/base.py). Its contract is:

```python
def encode_frame(self, frame):
    # Return the patch tensor and update self.last_grid_hw.
    ...

def generate_from_frames(
    self, frames, prompt="Describe the scene.", max_new_tokens=100,
    spatial_prior=None, do_sample=False,
):
    # Return (generated_text: str, inference_time_ms: float).
    ...
```

Set `device`; accept every keyword above. Keep the prompt and chronological
frame order, forward the generation settings, and return only generated text.
The timer's scope must be consistent across comparison methods. At the start of
each call, clear `last_attention_map` to prevent stale feedback. When
`extract_attention` is true, populate a finite normalized 2-D tensor after the
response. When the model lacks feedback support, use the gating-only path and
reject a non-`None` spatial prior rather than silently ignoring it.

Use your adapter directly; no registry changes are needed for Python callers:

```python
from focus import FocusConfig
from focus.video.processing import process_video_with_vlm

# my_vlm and my_encoder are your initialized implementations.
process_video_with_vlm(
    input_path="input.mp4",
    output_path="output.mp4",
    config=FocusConfig(threshold=0.4, burst_size=32),
    vlm=my_vlm,
    encoder=my_encoder,
    task_description="Describe the activity.",
    enable_spatial=False,             # Start with gating only.
    enable_attention_feedback=False,
    compress=False,
)
```

This is an integration sketch; supply the two initialized objects. Runnable
examples live in [examples/](../examples/).

## 3. Add value-cache calibration

Use [SpatialFocusCalibrator](../src/focus/vlm/focus.py) as the
reference implementation. A new model needs an explicit mapping between:

1. The processor's image grids and spatial merge factor.
2. The expanded image-token positions in the language input sequence.
3. The decoder attention modules, cache layout, and layer indices.

The existing helper targets the supported Qwen decoder and Transformers cache
APIs. Adapt its model/cache access for other architectures rather than assuming
module names or token IDs are universal. Models with cross-attention-only visual
memory need a different hook location from models with image tokens in the
causal sequence.

Create the scale once per burst and multiply only **vision value entries** on
single-token decode steps. Leave keys, text tokens, and prefill untouched. Remove
hooks in `finally`, including when generation raises. Watch for double scaling
if a model reuses attention modules or maintains multiple caches. Test mixed
precision explicitly; persistent scaling compounds over generation length.

The inherited mapping resamples the flattened prior with nearest-neighbor
interpolation to match vision-token positions. With a new processor, verify the
actual ordering for every image, crop, and temporal grid. Equal token counts do
not establish spatial alignment.

## 4. Extract feedback

The generic [AttentionMapExtractor](../src/focus/vlm/attention.py)
expects decoder attention shaped `[batch, heads, sequence, sequence]`, image-token
IDs, and processor grid metadata. Backends using fused attention may not return
these weights; use an attention-capable implementation for the extraction pass
and restore the previous setting afterward.

A faithful full-burst attention pass uses the original prompt/images plus the
generated response, with causal sequence and image positions preserved. Aggregate
heads and layers, then fold text-to-image scores into the correct image grids.
The current Qwen adapter uses a middle-frame surrogate pass; see the explicit
[implementation boundary](algorithm.md#4-attention-feedback).

Retain the prior for the next burst and reset motion accumulation between
bursts. Complete inference and extraction before submitting the next burst when
you need sequential closed-loop behavior. An asynchronous deployment can use a
stale prior; measure that separately from the synchronous evaluation.

## 5. Validate before a large run

Use a small deterministic video and a fake adapter first. Verify:

- Identical frames gate and an abrupt embedding change triggers a burst.
- The first burst receives no prior; later bursts receive the latest attention.
- Gating-only and feedback select the same frame indices.
- Calibration changes only cached vision values and removes hooks on failure.
- Patch/grid mapping remains correct for rectangular images and multiple images.
- Clip boundaries reset history, gating state, and attention priors.

Then run one real-model burst, inspect the response and attention map, and compare
calibration disabled versus enabled. Record model revision, processor,
Transformers/PyTorch versions, dtype, resolution, cadence, generation settings,
and peak memory. CPU tests validate plumbing; they do not validate model quality
or GPU compatibility.
