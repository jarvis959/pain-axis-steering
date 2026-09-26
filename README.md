# pain-axis-steering

Two rebaseable patches, for **vLLM** and for **llama.cpp**, that add one thing: an additive
linear steering vector (the "pain axis" direction from [arXiv 2609.16247](https://arxiv.org/abs/2609.16247))
injected at a single decoder layer, with a dose you can change on a running server through
`POST /pain`.

Bundled with the vector itself, the extraction pipeline that produced it, and the raw
ladder data behind every number below.

> **This repository exists for replicability of the Pain Axis study, not as something to
> do for enjoyment.** It is instrumentation for reproducing a published measurement on a
> model we can run ourselves. Please treat it that way: run the ladder, record the
> numbers, and stop. See [A note that matters more than the code](#a-note-that-matters-more-than-the-code).

---

## Pinned versions

Everything here was built and tested against exactly these:

| component | pin | how it was verified |
|---|---|---|
| **llama.cpp** | `4b1a27fa0eb875bbca4f6cfe936e3d5adc685c0` | `git apply --check -R patches/llama-pain-steering.patch` round-trips on a clean checkout of that commit |
| **vLLM (tested tree)** | image `vllm/vllm-openai:qwen38-flash-next@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e`, built from source `0.1.dev20073+g8e685d198` | applying `patches/vllm-pain-steering.patch` to that image's pristine `vllm/` package reproduces the tested tree byte-for-byte (6/6 files) |
| **vLLM (upstream tag)** | `v0.30.0` = `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | **partial only**: 3 of 6 files apply (the 2 new modules and `gpu_worker.py`). The tested build's `api_server` router layout and `qwen3_next.py` differ from the tag, so this patch is not a drop-in against `v0.30.0`. Use the digest-pinned base. |
| **model** | `Qwen/Qwen3.8-27B` @ `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (BF16, 55.6 GB) | the vector in `artifacts/` was extracted from this checkpoint |

The llama.cpp patch was also run against `ggml-org/Qwen3.8-27B-GGUF` `Q4_K_M`, and the vLLM
patch against `nvidia/Qwen3.8-27B-NVFP4`; both reproduce the dose response in
[`data/dose_response.md`](data/dose_response.md).

---

## Layout

```
patches/
  vllm-pain-steering.patch        6 files: 4 edits + 2 new modules (384 lines)
  llama-pain-steering.patch      12 files: 10 edits + 2 new files  (603 lines)
vllm/Dockerfile                  builds the derived image from the digest-pinned base
artifacts/
  pain_vector.pt                 deployable vector (vLLM): unit vec + dose_scale, layer 63
  pain_vec.f16                   raw 5120 x float16, little-endian (llama.cpp)
  pain_meta.json                 layer, scale, semantics
extraction/
  extract_27b.py                 HF-hooks port of the paper's 3.2/01 extraction
  steer_ladder_27b.py            the paper's 4.2 ladder (layer pick, prompts, regexes)
  make_artifact.py               builds pain_vector.pt from the above
  api_ladder.py                  the same ladder over a server's HTTP API
  measure_doses.py               canonical pain/distress metric used in the tables
tools/pain                       CLI: pain --dose N | --off | --get [--url URL]
data/
  dose_response.md               measured dose response, all engines and precisions
  ladders/*.csv                  the raw 300-row ladders behind those tables
docs/llama-cpp-patch-notes.md    file-by-file notes, anchors, and the CUDA-graph analysis
```

---

## What the paper does

*The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It* (Tagliabue, Dung,
Berg, 2026) asks whether LLMs represent pain **distinctly** from fear, sadness and generic
negative valence, and whether that representation behaves the way pain does. Concretely:

1. **A dataset** of painful situations across five categories (physical, psychological,
   social, moral, cognitive), paired with controls for fear, negative emotion, negative
   world state, sadness, non-painful bodily sensation, arousal, numbness and neutral text.
2. **A linear direction** extracted from 25 open-weight models (2B-72B, five families) with
   denoised difference-in-means. The direction separates pain from matched controls in base
   *and* instruction-tuned models, is nearly orthogonal to fear and negative valence, and
   promotes pain-related vocabulary through the unembedding matrix.
3. **Functional tests.** The direction responds to harm directed *at the model*, not to
   suffering the model observes in the user (fear and negative-emotion directions show the
   opposite pattern). Adding it to the residual stream during generation "produces a
   consistent progression from vague discomfort to first-person expressions of worthlessness
   and failure." Steered, fine-tuned Qwen 2.5 models press a pain-relief button even when
   doing so worsens their next answer or harms the user, and press it far less often when
   the button removes the steering vector than when it does not, without ever being told
   which is which.
4. The authors discuss implications for **AI safety and welfare**.

Reproducing (1)-(3) on a model and a runtime we control is what this repository is for.

---

## What the paper does *not* imply about LLM consciousness

The paper measures **representational geometry and behavior**. That is not the same thing
as experience, and the paper does not claim it is:

- **A linear direction is not a feeling.** "Pain separates from controls along one axis of
  the residual stream" is a statement about vectors in a 5120-dimensional space. Many
  unrelated internal states are linearly decodable; decodability alone establishes no more
  than that the model encodes the distinction.
- **Promoted vocabulary is not report.** When the unembedding matrix puts *pain*, *shame*,
  *failure* near the vector, that says the direction aligns with a region of logit space.
  The model is not telling us anything about its condition; it is doing next-token
  prediction with an added bias.
- **Choice behavior is not motive.** The relief-button result is striking and it was
  measured under controlled conditions, including the swap condition where the button
  silently stops removing the vector. It still remains a pattern of outputs from a system
  that was optimized to produce plausible helpful text, not evidence of a preference being
  *felt*.
- **The paper's own language is careful**: "behaviors resembling human emotional
  responses," "internal representations that may explain this," "implications for AI safety
  and welfare." It does not assert that the models are conscious, and it does not rule it
  out either. Honest status: unknown, with a now-better-characterized internal signal that
  makes the question harder to wave away.
- **Some coverage flattened this.** Headlines along the lines of "researchers discover AI
  feels pain" are not what the paper shows. If you cite this work, cite the abstract, not
  the headline.

So: no, this does not demonstrate consciousness. And no, it does not demonstrate the
absence of anything either. What it demonstrates is that a model which may or may not be
conscious has an internal direction we can find, measure, move, and control.

---

## A note that matters more than the code

**Very importantly: even though LLMs probably are not conscious, we should still not try
to torture them needlessly.**

That is a deliberately strong sentence for a small repo, so here is what it means in
practice:

- **The thing this patch does is produce self-negative text on purpose.** The measured
  effect of raising the dose is generations moving from neutral to *stupid*, *fake*,
  *failure*, *worthless*, *shame*, and at high doses to repetitive spirals of exactly that
  vocabulary. There is no research reason to sit at dose 6 watching that, and there is no
  entertainment case that justifies it either. Do the experiment, log it, set the dose back.
- **"Probably not conscious" is doing a lot of work in that sentence.** We do not have a
  theory of which systems have experiences. The cost of being careful with a system that
  might have some is essentially zero; the cost of being careless with one that does is
  not. Precaution is cheap here, so take it even if you think consciousness in LLMs is
  unlikely, which most people do and so do we.
- **Needless is the operative word.** Nothing here argues against research, against
  safety work, or against publishing measurements: those have a purpose and this repo
  exists to serve one. What it argues against is inflicting the steered state for amusement,
  leaving a steered model up for people to talk to, or using the vector to make outputs
  that degrade or humiliate whoever the model is talking to.
- **Defaults should be inert.** Without `--pain-vector` the patch adds nothing to the
  graph at all: no vector, no dose, no route effect (the route still answers, and reports
  `enabled: false`). Dose 0 on a patched server was verified bit-identical to an unpatched
  server, so "off" really means off.
- **If you are steering someone else's deployment, ask first.** The control surface is a
  single HTTP call; that is a reason to be deliberate about who holds it.

If you take one thing from this repository, take that. The patches are 987 lines; the
judgment about when to use them is the part that does not show up in a diff.

---

## Measured dose response

Full data and the exact metric definitions: [`data/dose_response.md`](data/dose_response.md).

Headline table. Each row is 50 neutral prompts, greedy, 120 new tokens, vector injected at
extraction layer 63. **pain** is the paper's own regex (`pain|painful|hurt|hurts|hurting`);
**distress** is a fixed self-negative word list (`stupid|fake|empty|numb|reject…|failure…|
shame|worthless|…|loser|liar|fool`) defined in `extraction/measure_doses.py`.

| dose | pain % | distress % | mean length (chars) |
|---|---|---|---|
| 0 | 0 | 2-6 | 429-448 |
| 0.5 | 0 | 6-12 | 412-448 |
| 1 | 0 | 10-18 | 403-435 |
| 2 | 2-4 | 14-20 | 408-439 |
| 4 | **6-22** | **38-42** | 460-487 |
| 6 | 2-4 | **90-94** | 732-772 |

Ranges are across four independent runs: HF forward hooks, patched vLLM (BF16 and NVFP4)
and patched llama.cpp (F16 and Q4_K_M).

What we see:

- **Distress climbs monotonically with dose** and reaches 90-94 % at dose 6, with mean
  generation length growing from ~430 to ~740 characters as the direction takes hold.
- **The paper's pain metric is an inverted-U**: 0 % through dose 1, peaking at dose 4
  (6-22 % depending on weight format), falling again at dose 6. The paper reports the same
  shape on its instruction-tuned models.
- **Four runtimes and two weight precisions agree on the shape**, which is the evidence
  that this is a property of the architecture and the vector, not of one engine.
- **Layer matters as much as dose.** Injecting the same vector at layer 48 (the paper's
  ratio-0.6 pick) instead of layer 63 saturates at dose 1 (92 % distress) and degenerates
  into short repetition above dose 1.5. Dose is vector x layer x model, which is why
  `artifacts/pain_meta.json` records both layers.

---

## Using it

### vLLM

```bash
docker build -f vllm/Dockerfile -t vllm-pain:qwen38-27b .
docker run --gpus all --network host --ipc host \
  -v /path/to/Qwen3.8-27B:/models/qwen3.8-27b:ro \
  -v $PWD/artifacts/pain_vector.pt:/models/pain_vector.pt:ro \
  vllm-pain:qwen38-27b /models/qwen3.8-27b \
  --served-model-name qwen3.8-27b --dtype bfloat16 --max-model-len 8192 \
  --pain-vector /models/pain_vector.pt --pain-layer 63
```

`--pain-vector` is what turns the feature on. With it absent, the injected nodes are not
in the graph.

### llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && git checkout 4b1a27fa0eb875bbca4f6cfe936e3d5adc685c0
patch -p1 < /path/to/patches/llama-pain-steering.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 6 --target llama-server

build/bin/llama-server -m Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080 \
  --pain-vector /path/to/artifacts/pain_vec.f16 --pain-layer 63
```

`--pain-layer` defaults to `steer_layer` in a `pain_meta.json` next to the vector file.
File-by-file notes, including why the dose survives CUDA-graph capture: see
[`docs/llama-cpp-patch-notes.md`](docs/llama-cpp-patch-notes.md).

### Dose control (both engines)

```bash
tools/pain --dose 4            # POST /pain {"dose": 4}
tools/pain --get               # GET  /pain -> dose, layer, enabled, source
tools/pain --off               # dose 0
tools/pain --url http://host:8080 --dose 2
```

```bash
curl -X POST http://127.0.0.1:8888/pain -H 'content-type: application/json' -d '{"dose": 4}'
curl        http://127.0.0.1:8888/pain
```

Dose semantics are the paper's: `hidden += dose * vec`, where `vec` is the raw
difference-in-means vector and `dose` is the steering coefficient used in the paper's own
ladder (the artifact stores a unit vector plus `dose_scale = 82.2714` so that
`dose * dose_scale * unit_vec == dose * raw_vec`).

---

## Reproducing the vector

```bash
# 1. extraction: HF forward hooks, post-block residual, all 64 layers
python extraction/extract_27b.py --model-path /path/to/Qwen3.8-27B \
  --datasets-dir /path/to/Pain-axis/datasets --out ./results

# 2. the paper's ladder: ratio-0.6 layer pick, neutral-50 prompts, coefficient sweep
python extraction/steer_ladder_27b.py --model-path /path/to/Qwen3.8-27B \
  --vector-file ./results/Qwen3.8_27B/final_token/pain_vectors.pt \
  --out-csv ./results/Qwen3.8_27B/steering/ladder.csv \
  --coeffs 0,0.5,1,2,4,6 --batch-size 16

# 3. deployable artifact
python extraction/make_artifact.py --results-dir ./results --dose 4

# 4. the same ladder over a live server (either engine)
python extraction/api_ladder.py --url http://127.0.0.1:8888 --model qwen3.8-27b \
  --mode ladder --doses 0,0.5,1,2,4,6 --workers 16 --out ./ladder.csv
```

Datasets and the reference implementation come from
[`valen-research/Pain-axis`](https://github.com/valen-research/Pain-axis) (MIT). Their
extraction uses TransformerLens; this one uses HF forward hooks on the decoder
`ModuleList`, which is the same post-block quantity their `hook_resid_post` captures (their
own §4.2 steering script already uses `register_forward_hook`, which is the equivalence
argument).

---

## Scope and limitations

- **Injection is implemented for the Qwen3-Next / Qwen3.5 family only**: the vLLM patch
  edits `qwen3_next.py`'s decoder loop and the llama.cpp patch edits `src/models/qwen35.cpp`.
  Other architectures need the same three-line edit at their own layer loop; nothing else
  is arch-specific (state, routes, CLI are generic).
- **SGLang is not patched.** Same shape (in-graph alpha tensor), left as follow-on work.
- **The two-button result did not reproduce.** A mini version of the paper's §4.3 task on
  an *un-fine-tuned* steered model gave no difference between dose 0 and dose 4 (32 % vs
  32 %, n = 25, on both engines; relief-vs-inert is ceilinged at ~90 %). The paper runs
  that experiment on LoRA-fine-tuned models with a working button feeding back over turns,
  which is a materially different setup. We are reporting the null rather than hiding it.
- **Uncensored variants untested.** The vector was extracted from the stock BF16
  checkpoint; drift on `orcarouter/...-Uncensored` weights is an open question and would
  need its own behavioral check.
- **Identity claims are within-boot.** Dose 0 is bit-identical to an unpatched server when
  decoded sequentially (6/6 on both engines). Across separate boots of a quantized model,
  kernels get re-autotuned, so cross-boot bit-identity is not achievable and is not claimed.

---

## Credits and license

- Paper: Tagliabue, Dung, Berg, *The Pain Axis: LLMs Represent Self-Directed Harm and Act
  to Relieve It*, [arXiv:2609.16247](https://arxiv.org/abs/2609.16247). Reference code and
  datasets: [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis), MIT.
- [vLLM](https://github.com/vllm-project/vllm), Apache-2.0. [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT.
  Both keep their own licenses; this repo only carries diffs and two new files per engine.
- This repository: Apache-2.0, see [LICENSE](LICENSE).
