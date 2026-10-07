# FOCUS project website

A static research website for *FOCUS: Closed-Loop Attention Feedback for Efficient Vision-Language Understanding*. The public site is [akul-nv.github.io/focus](https://akul-nv.github.io/focus/). No package installation or build step is required for the website.

This directory also contains the implementation guides: [the core method](algorithm.md), [baselines and evaluation](benchmarks.md), and [new VLM integration](new_vlm.md). For installation and running FOCUS, start with the [repository README](../README.md). The website links to these guides on GitHub.

## Local preview

From the repository root:

```sh
python3 -m http.server 8000 --directory docs
```

Open <http://localhost:8000>. Stop the server with `Ctrl+C`.

## Publish with GitHub Pages

The repository uses **Settings → Pages → Deploy from a branch**, with branch **main** and folder **/docs**. Pushing website changes to `main` publishes `docs/index.html` at <https://akul-nv.github.io/focus/> after deployment completes. The source folder name is not part of the published URL. A custom Actions workflow is not needed for this static site.

The site uses relative asset links so it works both at this repository URL and on a local preview server. `docs/index.html` inside this directory redirects the former `/focus/docs/` URL to `/focus/`, preserving query parameters and section anchors when JavaScript is enabled. The `.nojekyll` file keeps the site as plain static files; the Markdown implementation guides are intended to be read on GitHub.

## Sources and assets

The supplied paper and poster are the scientific sources for the website. Figures are copied from the supplied paper assets without modification. The original source files remain unchanged.

| Website asset | Supplied source |
| --- | --- |
| `assets/focus-paper.pdf` | `1746_FOCUS_Closed_Loop_Attention.pdf` |
| `assets/focus-poster.pdf` | `latex/poster.pdf` |
| `assets/convergence.png` | `paper_assets/images/convergence.png` |
| `assets/soup-frames.png` | `paper_assets/images/soup_frames.png` |
| `assets/sauce-frames.png` | `paper_assets/images/sauce_frames.png` |

Source paths in this table refer to the supplied poster workspace, outside the cloned repository. The downloadable copies and figures in `assets/` make the website self-contained.

When editing results, retain each benchmark's metric, baseline, backbone, and measurement conditions. VLM compute savings exclude the gating encoder unless explicitly identified as end-to-end; neither quantity alone establishes real-time throughput on arbitrary hardware. The interactive method illustration is explanatory and does not execute FOCUS or generate experimental measurements. Qualitative examples illustrate individual clips and should not be presented as aggregate benchmark results. Consult the full paper for experimental details and limitations.

The research implementation lives in [`src/focus`](../src/focus), with runnable examples, tests, and command-line entry points described in the repository README. The website's interactive figures display the paper's reported results; they do not run model inference.
