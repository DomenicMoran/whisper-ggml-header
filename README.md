# whisper-ggml-header

Read and validate the header of a Whisper GGML model. Catches the mis-converted
files that whisper.cpp silently refuses to load. Reads 48 bytes. No
dependencies.

```bash
# Not on npm yet, so npx needs the repository:
npx github:DomenicMoran/whisper-ggml-header model.bin
```

```
model.bin
  n_text_ctx    1024   <- wrong
  n_vocab       51865
  n_mels        80
  n_audio_ctx   1500
  n_text_layer  6
  ftype         2008 (q5_0 (qnt v2))
  BLOCKING n_text_ctx: n_text_ctx is 1024, the classic broken conversion.
           The script took max_length from config.json instead of
           max_target_positions. whisper.cpp will not load this file.
  => whisper.cpp will refuse this file
```

Exit code is 1 when any file would fail to load, so it works as a step in a
release pipeline.

## The bug it catches

whisper.cpp requires `n_text_ctx = 448`. Anything else and the model does not
load; bindings such as `whisper.rn` surface that as a generic "model
unavailable", with no mention of the header.

A large share of community conversions of fine-tuned Whisper models get this
wrong, and the reason is the same every time. The upstream conversion script,
`models/convert-h5-to-ggml.py`, writes `max_length` from the Hugging Face
`config.json` into the `n_text_ctx` slot. Several fine-tunes carry:

```json
"max_length": 1024,          // a generation parameter
"max_target_positions": 448  // the actual decoder context length
```

The right field is `max_target_positions`. If you are converting yourself,
patch the script:

```python
hparams["max_length"] = int(hparams.get("max_target_positions") or 448)
assert hparams["max_length"] == 448, hparams["max_length"]
```

And check the result before you ship it, which is what this package is for.

## Library

```ts
import { inspect, parseHeader, checkHeader } from "whisper-ggml-header";
import { open } from "node:fs/promises";

const file = await open("model.bin", "r");
const bytes = new Uint8Array(48);
await file.read(bytes, 0, 48, 0);
await file.close();

const { header, findings, loadable } = inspect(bytes);
if (!loadable) throw new Error(findings[0].message);
```

Findings are `blocking` or `warning`. Blocking means whisper.cpp will refuse
the file. Warning means the header is loadable but something does not match a
known Whisper configuration, which usually points at a mixed-up tokenizer or
mel filter bank.

| Check | Severity | Why |
| --- | --- | --- |
| `magic` is `0x67676d6c` | blocking | A GGUF file needs a different loader entirely |
| `n_text_ctx` is 448 | blocking | The one whisper.cpp enforces |
| `n_mels` is 80, or 128 for large-v3 | blocking | Mel filter bank does not match the model |
| `n_vocab` is 51864, 51865 or 51866 | warning | Anything else means the tokenizer got mixed up |
| `n_audio_ctx` is 1500 | warning | 30 seconds of audio, the Whisper window |
| `ftype` is a known type | warning | Unknown quantisation |

## Header layout

Twelve little-endian `int32` values, 48 bytes total:

| Offset | Field | Typical |
| --- | --- | --- |
| 0 | magic | `0x67676d6c` |
| 4 | n_vocab | 51865 |
| 8 | n_audio_ctx | 1500 |
| 12 | n_audio_state | 512 (base) |
| 16 | n_audio_head | 8 |
| 20 | n_audio_layer | 6 |
| **24** | **n_text_ctx** | **448** |
| 28 | n_text_state | 512 |
| 32 | n_text_head | 8 |
| 36 | n_text_layer | 6 |
| 40 | n_mels | 80 |
| 44 | ftype | see below |

### Reading `ftype`

Quantised files encode the type as `quantisationVersion * 1000 + baseType`.
`describeFtype` splits it:

```ts
describeFtype(2008); // { quantisationVersion: 2, baseType: 8, label: "q5_0 (qnt v2)" }
describeFtype(1);    // { quantisationVersion: 0, baseType: 1, label: "f16" }
```

One detail worth reading twice, because it is easy to get backwards: in the
`ggml_ftype` enum **7 is q8_0 and 8 is q5_0**, and 5 and 6 do not exist. This
was verified against real files rather than assumed: the same base conversion
quantised to q5_0 reports 2008 at 55 MB, and to q8_0 reports 2007 at 82 MB.

## Where it comes from

An on-device Quran recitation checker in a React Native app. The speech
recognition kept reporting itself as unavailable on real devices while
everything looked fine in the source tree. The model file had `n_text_ctx =
1024`, inherited from a third-party conversion whose provenance was not
documented.

The fix was to convert the original model in-house with a patched script. The
check in this package is what now runs before any model file is published, so
the same class of failure cannot reach a device again.

## Licence

MIT. See [LICENSE](LICENSE).
