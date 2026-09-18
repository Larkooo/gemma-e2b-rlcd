# Third-party notices

## MLX-VLM

This project integrates [MLX-VLM](https://github.com/Blaizzy/mlx-vlm). Its Gemma-specific adapters in `gemma_decisions/answer_positions.py` and `gemma_decisions/vision.py` follow and adapt the upstream Gemma implementation. The upstream license is retained below.

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

The typed-decision interface and RLCD research direction are inspired by [TypeSafe's public Jev description](https://typesafe.ai/blog/introducing-system-one-models-and-jev). No TypeSafe source code, proprietary training recipe, or model weights are included. The project is not affiliated with TypeSafe or Google.
