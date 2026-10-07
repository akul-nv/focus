# Running comparisons

`focus-benchmark` runs streaming captioning on local videos. `focus-egoschema`
adds a final multiple-choice QA call. Both use the same synchronous stream runner
and create fresh gate, caption history, and attention state for each video.
They accept one method per run so model ownership and output files are explicit.

## Methods retained from the source branch

| Method | Selection and generation behavior |
|---|---|
| `uniform` | Consecutive bursts at a fixed sampled-frame interval |
| `bolt` | The same scheduled bursts, then CLIP query scoring and BOLT inverse-CDF frame selection |
| `dycoke` | The same scheduled bursts, then embedding-based frame selection and decoder cache masking |
| `gating` | FOCUS temporal gate without attention feedback or calibration |
| `focus` | Temporal gate, spatial prior, value-cache calibration, and attention feedback |

**BOLT is a streaming adaptation.** Selection sees only the current burst, not
the complete video. The query is configurable with `--bolt-query`; it must be
available at inference time. The selected images feed the same generator used
by the other methods. See [the original BOLT implementation](https://github.com/sming256/BOLT)
and the [retained license notice](../THIRD_PARTY_NOTICES.md).

**DyCoke is an adaptation, not the upstream implementation.** It removes complete
frames and masks cached entries while keeping tensor shapes fixed. It does not
perform the original method's temporal token averaging or physical cache
compression, so do not infer equivalent memory or speed savings. Use
`--dycoke-no-cache` for the frame-selection component alone. See
[the original DyCoke implementation](https://github.com/KD-TAO/DyCoke).

## Captioning

Install `'.[qwen]'` as described in the README, then run:

```bash
focus-benchmark clip.mp4 --method focus --output results/focus.json \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --encoder-model Qwen/Qwen2-VL-2B-Instruct \
  --target-fps 24 --burst-size 32 --interval 100 \
  --threshold 0.4 --max-gate-frames 300 \
  --focus-alpha 1.05 --focus-range 0.75 1.0 --focus-suppress 0.04

focus-benchmark clip.mp4 --method uniform --output results/uniform.json
focus-benchmark clip.mp4 --method bolt --output results/bolt.json
focus-benchmark clip.mp4 --method dycoke --output results/dycoke.json
focus-benchmark clip.mp4 --method gating --output results/gating.json
```

`--interval` is measured in **sampled frames**, after the `--target-fps` step.
With 24 fps and interval 100, scheduled bursts start about every 4.17 seconds.
The interval must be at least the burst length; these runners do not create
overlapping bursts. Lower-rate inputs retain their actual source timing rather
than duplicating frames. A final partial burst is processed.

All methods default to greedy generation with a 100-token cap. `--do-sample`
enables sampling and `--seed` sets its seed. Each continuation sees the previous
caption, truncated to 80 characters. Uniform, BOLT, and DyCoke share the fixed
schedule; gating and FOCUS share identical semantic gate settings. Selection
inside BOLT/DyCoke can reduce the number of frames sent per call.

Useful baseline parameters:

| Option | Default | Meaning |
|---|---:|---|
| `--bolt-frames` | 16 | Number of selected frames per burst |
| `--bolt-power` | 3 | Contrast exponent for CLIP scores |
| `--clip-model` | `openai/clip-vit-large-patch14` | BOLT scoring model |
| `--dycoke-drop-fraction` | 0.3 | Temporal frame reduction target |
| `--dycoke-keep-fraction` | 0.8 | Cache retention fraction |
| `--dycoke-start-layer` | 3 | First decoder layer with cache masking |
| `--dycoke-similarity-threshold` | 0.9 | Attention-change threshold for cache suppression |

Outputs include run arguments, captions, burst source frame indices, selected
frame indices, timestamps, counts, and measured timing. Existing output paths
are protected unless `--overwrite` is provided. Multiple inputs are supported:

```bash
focus-benchmark clip1.mp4 clip2.mp4 --method focus --output results/clips.json
```

The output is checkpointed atomically after each video; a partial file records
completed videos only. There is no implicit resume or reuse of prior outputs.

## EgoSchema-style multiple-choice QA

Obtain the videos and question/answer metadata separately from the
[EgoSchema project](https://egoschema.github.io/). No videos, annotations, or
model-generated ground truth are bundled. Convert your local metadata into a
JSON array, for example:

```json
[
  {
    "id": "example-clip",
    "video": "example-clip.mp4",
    "question": "What does the person do after opening the drawer?",
    "options": [
      "Picks up a spoon",
      "Closes the window",
      "Moves a chair",
      "Washes a plate",
      "Turns off the light"
    ],
    "answer": 0
  }
]
```

This is a **synthetic format example**, not an EgoSchema annotation. `video` is
resolved relative to `--video-dir`. There must be exactly five options;
`answer` is an optional zero-based index from 0 through 4. Omit it for unlabelled
inference. Keep your exact subset manifest and labels with your experiment.

```bash
focus-egoschema --questions data/questions.json --video-dir data/videos \
  --method focus --output results/egoschema-focus.json \
  --target-fps 6 --burst-size 32 --interval 100
```

Change `--method` and output path for each baseline. The final QA prompt combines
the caption narrative, question, five options, and up to `--max-qa-frames 16`
representative frames. Report caption calls and the extra QA call separately.
Accuracy uses only labelled questions as its denominator; unparseable answers
count as incorrect. Unlabelled runs must not be reported as measured accuracy.

## What this release can reproduce

The commands reproduce a documented, configurable comparison of the retained
source implementations. They do **not** promise the paper's exact scores:

- The source branch supplied a FOCUS EgoSchema runner but no complete matched
  baseline EgoSchema campaign or five-run seed/manifest record.
- The paper describes 6 fps FOCUS input, 32-frame bursts, and a matched 3.3-second
  cadence. A nonoverlapping 32-frame burst at 6 fps lasts 5.33 seconds, so those
  settings do not uniquely determine the described schedule. This runner makes
  cadence explicit and rejects overlapping fixed bursts.
- The captioning experiments used differing cadence and decoding settings for
  some baselines. This release defaults to one shared schedule/decoding policy;
  record overrides when reconstructing a specific experiment.
- Exact captioning clip IDs and judge prompts are not included. These commands
  produce captions and runtime metrics, not automated caption-quality scores.
- The shared evaluation prompt retains the latest caption truncated to 80
  characters. The paper describes a five-response history window.
- The current Qwen attention pass uses the middle frame, whereas the paper
  describes a full-burst prompt-and-response pass. See [algorithm notes](algorithm.md).
- Built-in adapters inherit the source dtype choice: FP16 on CUDA and FP32 on
  CPU. The paper lists BF16. Model quality and timing need validation for the
  actual dtype, hardware, dependency versions, and checkpoint revision used.

`wall_time_s` and pipeline-reported time have different scopes. Pipeline timing
may include selection and feedback extraction; the wall timer also includes
frame decoding and gating. Neither should silently be relabelled as the paper's
VLM-only `rho`. The published `rho` definition excludes the gating encoder and
BOLT's CLIP encoder. Retain full timing scopes alongside invocation counts.

For a new model, initialize your adapter and call `run_video`/`run_stream`
programmatically instead of relying on Qwen model auto-detection. See
[new VLM integration](new_vlm.md). GPU model runs and paper-table accuracy remain
separate from the CPU tests shipped here.
