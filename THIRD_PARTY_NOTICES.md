# Third-party notices

The FOCUS code is distributed under the project's [Apache-2.0 license](LICENSE).
The following notice also applies to the BOLT-derived frame-selection function.

## BOLT

The `inverse_transform_sampling` implementation retained in the BOLT baseline
was adapted from [`select_frames.py`](https://github.com/sming256/BOLT/blob/main/select_frames.py)
in the [BOLT repository](https://github.com/sming256/BOLT).
The adaptation adds validation and handling for degenerate score distributions;
the upstream MIT license and copyright notice are reproduced below.

Reference: Shuming Liu, Chen Zhao, Tianqi Xu, and Bernard Ghanem.
[BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding](https://arxiv.org/abs/2503.21483).
CVPR 2025.

### BOLT license

```text
MIT License

Copyright (c) 2025 BOLT Project Contributors

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
```

## DyCoke

The DyCoke baseline in this repository is a Qwen adaptation of ideas described
by Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang in
[DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models](https://arxiv.org/abs/2411.15024),
CVPR 2025. It uses frame selection and cache masking and is not the authors'
original implementation of token merging and cache compression.

The [official implementation](https://github.com/KD-TAO/DyCoke) is distributed
under [Apache-2.0](https://github.com/KD-TAO/DyCoke/blob/main/LICENSE).
This reference credits the method; it does not identify local files as copied
from the upstream implementation.
