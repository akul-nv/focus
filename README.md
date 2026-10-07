# FOCUS

**FOCUS: Closed-Loop Attention Feedback for Efficient Vision-Language Understanding**

The project website presents the method, results, qualitative examples, paper, and COLM poster. FOCUS connects semantic frame gating, spatial value-cache calibration, and decoder-attention feedback for streaming vision-language understanding without additional training.

## Project website

The website lives in [`docs/`](docs/) and uses static HTML, CSS, and JavaScript with no build dependencies.

Preview it from the repository root:

```sh
python3 -m http.server 8000 --directory docs
```

Open <http://localhost:8000>.

To publish with GitHub Pages, push the website to `main`, then select **Settings → Pages → Build and deployment → Deploy from a branch**, branch **main**, folder **/docs**. The intended URL is <https://akul-nv.github.io/focus/>; it becomes available after Pages is configured and deployment completes.

See [`docs/README.md`](docs/README.md) for file organization and source provenance.

## Research code coming soon

The research implementation has not been released yet. The website's interactive illustrations explain the method; they do not run the vision-language model.
