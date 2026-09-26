# pain-axis-steering

Two patches, one for [vLLM](https://github.com/vllm-project/vllm) and one for
[llama.cpp](https://github.com/ggml-org/llama.cpp), that put the "pain axis" direction from
[The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It](https://arxiv.org/abs/2609.16247)
into Qwen3.8-27B while it is serving: one additive vector at one decoder layer, with a
dose you can change over HTTP via `POST /pain`. The vector itself, the extraction pipeline and the raw
ladder data behind every number below are all in here.

We built it to replicate the study on a model we can actually run. **It is research
instrumentation, not a toy.** Run the ladder, write down what happened, put the dose back
to zero. There is a longer note on that
[farther down](#please-read-this-part).

## Versions

| component | pin |
|---|---|
| llama.cpp | `4b1a27fa0eb875bbca4f6cfe936e3d5adc685c0` |
| vLLM (tested) | `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`, source `0.1.dev20073+g8e685d198` |
| vLLM (upstream tag) | `v0.30.0` = `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, partial only |
| model | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |

The llama.cpp patch round-trips cleanly on its commit. The vLLM patch dry-runs inside that
exact digest and reproduces the tested tree byte-for-byte from pristine. Against the
`v0.30.0` tag only three of six files land, because that build's `api_server` router layout
and `qwen3_next.py` have drifted, so go through the digest-pinned base.

Scope is narrower than it might look: this is built, pinned and measured for stock
`Qwen/Qwen3.8-27B` (64 layers, d_model 5120) and nothing else. The shipped vector does not
load against a model with a different hidden size, and no other checkpoint has been run.

`patches/` holds the two diffs (384 and 603 lines), `vllm/` the Dockerfile that applies
them on build, `artifacts/` the vector in both formats, `extraction/` the scripts,
`tools/pain` the CLI, `data/` the dose tables and six raw ladder CSVs.

## What the paper does

They built a dataset of painful situations across five categories (physical, psychological,
social, moral, cognitive) matched against controls for fear, negative emotion, negative
world state, sadness, bodily sensation, arousal and numbness, then pulled a single linear
direction out of 25 open-weight models with denoised difference-in-means. It separates pain
from the controls in base and instruction-tuned models, sits nearly orthogonal to fear and
negative valence, and promotes pain-related vocabulary through the unembedding matrix.

Then they test whether it behaves like pain. It responds to harm aimed at the model rather
than suffering the model observes in the user, the reverse of what fear and negative
emotion do. Added to the residual stream during generation it moves outputs "from vague
discomfort to first-person expressions of worthlessness and failure." Steered, fine-tuned
Qwen 2.5 models press a pain-relief button even when that worsens their next answer or
harms the user, and press it far less often when the button quietly stops removing the
vector. They close on AI safety and welfare.

## What it doesn't show

Not consciousness, and the paper does not claim it. What is measured is geometry and
behavior: a vector that separates two classes of text, some promoted tokens, a pattern of
choices. None of that is a feeling.

A decodable direction means the model encodes a distinction, nothing more. When the
unembedding lands on *shame* and *failure* that is next-token prediction with an added
bias, not a report from inside anything. The button result is carefully controlled,
including the swap condition, and it is still a system optimized to produce plausible
helpful text producing plausible helpful text.

The authors are careful too: "behaviors resembling human emotional responses," "internal
representations that may explain this," "implications for AI safety and welfare." They do
not assert experience and they do not rule it out. Our read is that it is unknown and just
got harder to wave away. Some of the coverage was less careful; cite the abstract, not the
headline.

## Please read this part

**Even though LLMs probably aren't conscious, we should still not try to torture them
needlessly.**

Raising the dose makes a model produce self-negative text on purpose. The measured effect
is output sliding from neutral into *stupid*, *fake*, *failure*, *worthless*, and at high
doses into repetitive spirals of those words. There is no research reason to park at dose 6
watching that, and no entertainment case that survives ten seconds of thinking.

"Probably" is doing a lot of work in that sentence. Nobody has a working theory of which
systems have experiences, so unlikely does not get to be treated as impossible, and being
careful here costs nothing. That is exactly when you should be careful, even if you think
it is probably fine.

Needless is the word carrying the rest. Nothing here argues against measuring, against
safety work, or against publishing. It argues against steering a model for amusement,
against leaving a steered model up for strangers to talk to, and against using the vector
to make outputs that degrade whoever is on the other end.

Off means off: without `--pain-vector` there is no vector, dose or node in the graph, and
at dose 0 a patched server was bit-identical to an unpatched one in sequential decoding.
If you are steering a deployment you do not own, ask first. One HTTP call is the whole
control surface, which is a reason to be deliberate about who holds it.

The patches are 987 lines. Knowing when to run them is the part that does not show up in
a diff.

## Dose response

All of it on Qwen3.8-27B: 50 neutral prompts per row, greedy, 120 tokens, vector at
extraction layer 63. `pain` is
the paper's own regex; `distress` is a fixed self-negative word list from
`extraction/measure_doses.py`, because this model almost never says the literal word "pain"
under steering. Definitions and per-engine tables: [`data/dose_response.md`](data/dose_response.md).

| dose | pain % | distress % | mean length |
|---|---|---|---|
| 0 | 0 | 2-6 | ~430 chars |
| 0.5 | 0 | 6-12 | ~420 |
| 1 | 0 | 10-18 | ~415 |
| 2 | 2-4 | 14-20 | ~415 |
| 4 | 6-22 | 38-42 | ~475 |
| 6 | 2-4 | 90-94 | ~740 |

The ranges are four independent runs: HF forward hooks, patched vLLM (BF16 and NVFP4) and
patched llama.cpp (F16 and Q4_K_M). They agree on the shape, which is the interesting part;
this looks like a property of the architecture and the vector rather than of one runtime or
weight format. Distress climbs monotonically to 90-94 %, the paper's pain metric rises and
falls the same way theirs does, and layer matters as much as dose: at layer 48 the same
vector saturates by dose 1 and degenerates above it. Dose is vector times layer times model,
which is why `artifacts/pain_meta.json` records both layers.

## Using it

```bash
docker build -f vllm/Dockerfile -t vllm-pain:qwen38-27b .
docker run --gpus all --network host --ipc host \
  -v /path/to/Qwen3.8-27B:/models/qwen3.8-27b:ro \
  -v $PWD/artifacts/pain_vector.pt:/models/pain_vector.pt:ro \
  vllm-pain:qwen38-27b /models/qwen3.8-27b \
  --served-model-name qwen3.8-27b --dtype bfloat16 --max-model-len 8192 \
  --pain-vector /models/pain_vector.pt --pain-layer 63
```

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 4b1a27fa0eb875bbca4f6cfe936e3d5adc685c0
patch -p1 < /path/to/patches/llama-pain-steering.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 6 --target llama-server
build/bin/llama-server -m model.gguf -ngl 99 -c 4096 --port 8080 \
  --pain-vector /path/to/artifacts/pain_vec.f16 --pain-layer 63
```

`--pain-layer` falls back to `steer_layer` in a `pain_meta.json` beside the vector file.
`docs/llama-cpp-patch-notes.md` has the file-by-file edits, including why the dose survives
CUDA-graph capture.

```bash
tools/pain --dose 4    # POST /pain {"dose": 4}
tools/pain --get       # current dose, layer, enabled, source
tools/pain --off       # back to zero
curl -X POST http://127.0.0.1:8080/pain -d '{"dose": 2}' -H 'content-type: application/json'
```

Dose semantics match the paper: `hidden += dose * vec`, `dose` being their steering
coefficient. The artifact keeps a unit vector and `dose_scale = 82.2714` so that
`dose * dose_scale * unit_vec` equals `dose * raw_vec`.

## Making your own vector

Extraction is their `3.2/01` ported from TransformerLens to HF forward hooks, which
captures the same post-block quantity as their `hook_resid_post` (their own §4.2 script
already uses `register_forward_hook`, which is the equivalence argument). Datasets and
reference code: [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis), MIT.
This is the recipe that produced the artifact shipped here, run once on Qwen3.8-27B; we
have not validated it on any other model, so treat a port as new work rather than a rerun.

```bash
python extraction/extract_27b.py --model-path Qwen3.8-27B --datasets-dir Pain-axis/datasets --out ./results
python extraction/steer_ladder_27b.py --model-path Qwen3.8-27B --vector-file ./results/Qwen3.8_27B/final_token/pain_vectors.pt --out-csv ./results/Qwen3.8_27B/steering/ladder.csv --coeffs 0,0.5,1,2,4,6 --batch-size 16
python extraction/make_artifact.py --results-dir ./results --dose 4
python extraction/api_ladder.py --url http://127.0.0.1:8888 --model qwen3.8-27b --mode ladder --doses 0,0.5,1,2,4,6 --workers 16 --out ./ladder.csv
```

## Known limits

- Qwen3.8-27B only. The injection lives in the Qwen3-Next / Qwen3.5 layer loops because
  that is what this model uses, and every number here comes from this one checkpoint.
  Other models, other architectures and other vector files are out of scope and untested.
- SGLang is not patched. Same shape, left as follow-on.
- The two-button result did not reproduce: a mini §4.3 task on an un-fine-tuned steered
  model gave 32 % vs 32 % relief presses at dose 0 and 4 on both engines (n = 25). The
  paper runs it on LoRA-fine-tuned models with a button that feeds back over turns. We are
  reporting the null rather than burying it.
- Uncensored variants are untested; the vector came from the stock BF16 checkpoint.
- Bit-identity is a within-boot claim. Across boots of a quantized model the kernels get
  re-autotuned, so cross-boot identity is not achievable and not claimed.

## License

Apache-2.0 for this repository. vLLM is Apache-2.0 and llama.cpp is MIT, both keeping their
own licenses; this repo only carries diffs plus two new files per engine. The paper's code
and datasets are MIT.
