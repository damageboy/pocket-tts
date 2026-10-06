# Third-party notices

## pocket-tts-timestamped

The timestamp functionality is derived from
[dpm63/pocket-tts-timestamped](https://github.com/dpm63/pocket-tts-timestamped),
pinned to [revision 36e72b2](https://github.com/dpm63/pocket-tts-timestamped/commit/36e72b29d346c427c415acda99ce7118f37a541e).

The ported scope is the attention-based word-timestamp approach: selected
FlowLM attention-head capture and reduction to text-unit scores, source-word
and token/text-unit mapping, and streaming word-boundary alignment with voiced
audio gating and word-start/word-end events. The reference sources are
`pocket_tts_timestamped/timestamps/{alignment,text,records}.py` and the
timestamp integration in `models/tts_model.py` and `modules/transformer.py`.
The checkpoint-specific `timestamp_heads` selections in matching YAML configs
are also derived from that revision's `pocket_tts_timestamped/config/`.
Those selections are represented as zero-based
`(layer, head)` pairs; selections are copied only when both pinned weight paths
and model definitions match. They are not inferred from transformer dimensions.

This notice covers the derived software algorithms and configuration only.
Model weights, tokenizers, voice embeddings, and voice/audio assets retain
their separate upstream licenses and applicable usage terms. The software MIT
license below does not relicense those assets or grant additional model or
voice rights.

### MIT license

The following is the verbatim contents of the reference repository's `LICENSE`
at the pinned revision. That file does not contain a copyright line.

```text
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```
