# Third-party notices

## MLX-VLM

This project integrates [MLX-VLM](https://github.com/Blaizzy/mlx-vlm). Its Gemma-specific adapters in `gemma_rlcd/answer_positions.py` and `gemma_rlcd/vision.py` follow and adapt the upstream Gemma implementation. The upstream license is retained below.

MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Gemma weights

This repository does not redistribute Gemma weights or trained pilot weights. Download the model separately and consult the terms supplied with the selected revision:

- [Google Gemma 4 E2B model card](https://huggingface.co/google/gemma-4-E2B-it)
- [MLX Community conversion used by this project](https://huggingface.co/mlx-community/gemma-4-e2b-it-4bit)

The project's MIT license applies to project code and synthetic examples, not third-party model weights. `model-source.json` records the conversion and revision used for the archived local experiments.

## Research attribution

### Public parallel-decoding demo fixtures

`examples/demo-workloads.json` and the support-ticket and catalog portions of `gemma_rlcd/static/demo-presets.json` adapt public example data from [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), revision `2af86848be75847ccb3553b0941cc51d6ef7e4e9`, whose model card declares Apache-2.0. The license text is retained in [examples/UPSTREAM-APACHE-2.0.txt](examples/UPSTREAM-APACHE-2.0.txt). Original state and candidate labels are retained; schemas are translated to this project's contracts, some variants select subsets, and development expectations/incident propositions are added. The customer-inbox fixture is original synthetic data. Each adapted fixture records its source URL and hash. No upstream engine code or weights are incorporated.

## Demo typography

The demo video and poster use IBM Plex Sans, licensed under the SIL Open Font License. The license is retained in [docs/assets/Plex-OFL.txt](docs/assets/Plex-OFL.txt).

## Visual demo photograph

`gemma_rlcd/static/sample-street.jpg` is “Times Square (New York City)” by ISO Legacy, from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Times_Square_(New_York_City).jpg), dedicated under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). The original photograph is included without modification.
