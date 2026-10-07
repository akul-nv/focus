# FOCUS

**Closed-Loop Attention Feedback for Efficient Vision-Language Understanding**

FOCUS watches a video stream with a small vision encoder, invokes a larger VLM
when patch embeddings change, and uses the VLM's attention as spatial feedback
for the next burst. It combines **temporal gating**, **value-cache calibration**,
and **attention feedback** without training or removing visual tokens.

This repository contains the core method and the **Uniform, BOLT, and DyCoke**
comparison implementations. It includes a gating-only ablation, local-video
captioning, and an EgoSchema-style multiple-choice evaluator. Generated benchmark
outputs, datasets, LaTeX sources, and cluster scripts are excluded. The project
website retains its published figures, paper, and poster.

## Project website

Visit [the FOCUS website](https://akul-nv.github.io/focus/). Its static files live
in `docs/`; GitHub Pages publishes that directory at the project root URL.
The old `/focus/docs/` URL redirects to the root. To preview locally:

```bash
python3 -m http.server 8000 --directory docs
```

See [website maintenance](docs/README.md) for deployment and asset details.

## Install

Use Python 3.10–3.12. For actual Qwen inference, use a CUDA-capable machine with
memory for both the encoder and generator. The evaluation commands default to
`Qwen/Qwen2-VL-2B-Instruct` for encoding and `Qwen/Qwen3-VL-8B-Instruct` for
generation; weights download from
Hugging Face on first use. BOLT also downloads `openai/clip-vit-large-patch14`.
Model memory depends on image resolution, burst length, and attention extraction.
Choose a CUDA wheel with the [PyTorch install selector](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/akul-nv/focus.git
cd focus
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch/torchvision builds appropriate for your CUDA platform first.
python -m pip install -e '.[qwen,dev]'
```

Transformers is pinned to `4.57.6` because attention and cache APIs matter to
FOCUS. For CPU logic tests without model dependencies, install `'.[dev]'` instead.
With uv, the equivalent is `uv venv --python 3.11` followed by
`uv pip install -e '.[qwen,dev]'`. Optional FFmpeg enables video recompression;
use `--no-compress` to write the OpenCV output directly.

## Quick check without a GPU or model downloads

```bash
python examples/basic_usage.py
pytest -q
focus-gate --help
focus-video --help
focus-benchmark --help
focus-egoschema --help
```

The example exercises the gate on synthetic embeddings. Tests cover the
selection and feedback contracts; they do not establish checkpoint accuracy.

## Run FOCUS on a video

Generate a small input, or replace it with your own video:

```bash
python scripts/generate_test_video.py --output input.mp4

focus-video input.mp4 --output output.mp4 \
  --embed-model Qwen/Qwen2-VL-2B-Instruct \
  --inference-model Qwen/Qwen3-VL-8B-Instruct \
  --task 'Describe the activity.' \
  --threshold 0.4 --burst-size 32 \
  --focus-alpha 1.05 --focus-range 0.75-1.0 --focus-suppress 0.04 \
  --no-compress
```

This writes an annotated video and a text log beside it. Bursts run sequentially,
so each response supplies feedback before the next burst. Feedback is enabled by
default. Use `--no-spatial --no-attention-feedback` for gating only. The smaller
encoder decides **when** to invoke the generator; decoder attention from the
generator supplies the feedback. `focus-gate` runs temporal gating without generation.
All commands expose their options through `--help`.

For comparisons, use the synchronous evaluators below, which save structured
outputs and apply a shared sampling protocol.

## Compare the baselines

```bash
focus-benchmark input.mp4 --method focus --output results/focus.json
focus-benchmark input.mp4 --method gating --output results/gating.json
focus-benchmark input.mp4 --method uniform --output results/uniform.json
focus-benchmark input.mp4 --method bolt --output results/bolt.json
focus-benchmark input.mp4 --method dycoke --output results/dycoke.json
```

Each command runs one method and records its settings, captions, selected frame
indices, and timing. Keep input cadence, model, generation options, and burst
size fixed when comparing methods. See [benchmark instructions](docs/benchmarks.md)
for arguments, EgoSchema data format, baseline adaptations, and protocol limits.

**Scope:** these runners make the retained implementations usable. The source
branch does not contain everything needed to reproduce the paper's tables, and
some source behavior differs from the paper. Those differences are documented
in [the algorithm guide](docs/algorithm.md) and
[benchmark notes](docs/benchmarks.md); no recorded scores are bundled.

## Use a new VLM

Start with [the integration guide](docs/new_vlm.md). A `BaseEncoder` supplies
patch embeddings; a `BaseVLMPipeline` generates text from a burst. Full feedback
also requires access to decoder attention, visual-token positions, and mutable
value caches. A hosted VLM accepting images and returning only text can consume
gated bursts, but cannot provide this calibration loop.

## Code map

| Path | Purpose |
|---|---|
| `src/focus/gating.py` | Temporal capture/watch state machine |
| `src/focus/vlm/spatial.py` | Motion and attention fusion |
| `src/focus/vlm/focus.py` | Value-cache calibration hooks |
| `src/focus/vlm/attention.py` | Decoder attention extraction |
| `src/focus/vlm/qwen*_vl.py` | Qwen adapters |
| `src/focus/baselines/` | BOLT and DyCoke adaptations |
| `src/focus/evaluation.py` | Shared streaming comparisons |
| `src/focus/egoschema.py` | Multiple-choice QA evaluation |
| `examples/` | Minimal Python usage and extension examples |
| `tests/` | CPU regression tests |

The Python import is `focus`; the installable distribution is `focus-vlm`.
Use `FocusConfig` and `FocusController` for the Python API, `focus-gate` for
temporal gating, and `focus-video` for video generation with feedback.

## Development

```bash
python -m pip install -e '.[qwen,dev]'
ruff check .
pytest -q
python -m build
```

CI checks CPU behavior, command help, and packaging. A GPU smoke run also
validated FOCUS, gating-only, Uniform, BOLT, and DyCoke with Qwen2-VL-2B and
Qwen3-VL-8B on an RTX PRO 6000 Blackwell (Python 3.12, PyTorch 2.10/CUDA 12.8,
Transformers 4.57.6). It checked real decoding, attention feedback, cache
calibration, and cleanup on short synthetic input. This validates execution;
it does not reproduce the paper's benchmark accuracy or performance.

## License and attribution

FOCUS is licensed under [Apache-2.0](LICENSE). BOLT selection code carries the
upstream MIT notice in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), which also
lists the baseline references. Model weights and datasets have their own terms
and are obtained separately.
