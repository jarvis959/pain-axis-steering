# llama.cpp pain-axis steering patch - notes

Local patch: injects one fixed additive steering vector at one decoder layer with a
runtime-mutable dose, plus `GET`/`POST /pain` control routes in llama-server.
Semantics mirror the vLLM patch (`~/pain-axis/patches/vllm-pain-steering.patch`,
`~/pain-axis/vllm_build/vllm/pain_axis.py`).

- patch: `~/pain-axis/patches/llama-pain-steering.patch` (603 lines, 12 files, new files staged with `git add -N`)
- pin: `4b1a27fa0eb875bbca4f6cfe936e3d65adc685c0` (working tree intentionally left dirty, nothing committed)
- build: `cmake --build ~/pain-axis/llama.cpp/build -j 6 --target llama-server` → `[100%] Built target llama-server`
- verify patch round-trip: `git apply --check -R ~/pain-axis/patches/llama-pain-steering.patch` → OK

## Files touched (line anchors, post-patch)

| file | anchor | what |
|---|---|---|
| `include/llama.h` | 1667-1699 | public C API: `llama_pain_configure/enabled/layer/n_embd/vector_path/get_dose/set_dose` |
| `src/llama-pain.h` | 1-17 (new) | internal accessor `llama_pain_copy_scaled(float*, int)` |
| `src/llama-pain.cpp` | 1-231 (new) | state owner: f16 load, pain_meta.json fallback, mutex-guarded `scaled[]` |
| `src/CMakeLists.txt` | 39 | `llama-pain.cpp` added to `LLAMA_CORE_SOURCES` |
| `src/llama-graph.h` | 160-180 | `class llm_graph_input_pain` (graph input carrying the scaled vector) |
| `src/llama-graph.cpp` | 5, 128-147 | include + `set_input()` (per-step upload) + `can_reuse()` |
| `src/models/qwen35.cpp` | 154-171, 220-225 | input creation (guarded) + the `ggml_add` injection node |
| `common/common.h` | 691-693 | `common_params.pain_vector`, `.pain_layer` |
| `common/arg.cpp` | 1307-1316, 4747-4764 | configure-after-parse hook + `--pain-vector` / `--pain-layer` options |
| `tools/server/server-context.h` | 137-138 | `get_pain` / `post_pain` handler members |
| `tools/server/server-context.cpp` | 4828-4896 | both handlers (registered in `server_routes::init_routes`) |
| `tools/server/server.cpp` | 256-257 | `ctx_http.get/post("/pain", ...)` route registration (same style as `/props`) |

## Where the vector lands in the residual stream

Graph builder: `llama_model_qwen35::graph::graph()` in `src/models/qwen35.cpp`.

- `ggml_tensor * inpL` is the residual stream fed into each decoder block; the block body
  starts with `cur = build_norm(inpL, model.layers[il].attn_norm, ...)`.
- at the end of block `il`: `cur = ggml_add(ctx0, cur /*ffn_out*/, ffn_residual)` →
  `cur = build_cvec(cur, il)` - this `cur` is the **post-block residual of layer `il`**
  (the TransformerLens `hook_resid_post` quantity at layer `il`), and it becomes
  `inpL = cur` for layer `il+1`.
- injection (only when `--pain-vector` is set):
  `if (inp_pain && il == ll_pain) { cur = ggml_add(ctx0, cur, inp_pain->vec); cb(cur, "pain_inject", il); }`
  placed immediately before `inpL = cur` - so the addition happens *after* layer 48 has
  fully completed and *before* layer 49's input layernorm consumes it.
  For `--pain-layer = n_layer-1` the same code adds before the final `output_norm`.
- zero footprint when disabled: `llm_graph_input_pain` is only created if
  `llama_pain_enabled()`, so no input, no node, no `cb` entry → the built graph is
  identical to the unpatched build. Two `GGML_ASSERT`s guard layer range and
  `n_embd == 5120` at graph-build time (hard failure rather than silent no-steer).

## Dose semantics

- `vec` = raw f16 values from `pain_vec.f16` (5120 LE float16, norm 82.2714), converted to
  f32 once at configure time; `pain_vec_unit.f16` is *not* used.
- `residual += dose * vec`, dose initialized to **1.0** ⇒ dose 1.0 == the paper's steering
  coefficient (matches `pain_meta.json` `semantics` / `dose_default`).
- `POST /pain` → `llama_pain_set_dose()` recomputes `scaled[i] = dose * vec[i]` on the host
  under `g_mutex`; `llm_graph_input_pain::set_input()` copies `scaled` out and uploads it
  with `ggml_backend_tensor_set()` on every decode step → effective on the next token,
  **no graph rebuild, no recapture**.
- thread-safety: all state behind one `std::mutex`; the HTTP thread only writes, the
  inference thread reads a snapshot into its own per-graph staging buffer, then copies to
  the device outside the lock.

## CLI flags (evidence: `llama-server --help`)

```
--pain-vector FNAME                     path to a raw .f16 file (n_embd little-endian float16 values) with the
                                        additive steering vector
                                        the residual stream after the --pain-layer decoder block becomes
                                        `residual + dose * vector`, with the
                                        dose starting at 1.0 and controllable at runtime via GET/POST /pain
                                        (arXiv 2609.16247)
                                        (env: PAIN_VECTOR)
--pain-layer N                          0-based decoder layer that receives the steering vector
                                        (default: `steer_layer` from the pain_meta.json next to the
                                        --pain-vector file)
                                        (env: PAIN_LAYER)
```

Parse-time behaviour (all verified, exit code 1, no GPU touched):

- no `--pain-layer`, sibling `pain_meta.json` present → layer 48 taken from
  `"steer_layer"` (log: `pain steering: loaded .../pain_vec.f16 (5120 values, layer 48, dose 1.000)`);
  visible only with `-lv 4`, see Logging below.
- `--pain-layer 3` → overrides the meta (`... layer 3 ...`).
- no `--pain-layer` and no `pain_meta.json` → `pain steering: no --pain-layer given and no "steer_layer" found in '<dir>/pain_meta.json'`
- missing file → `pain steering: cannot open --pain-vector file '...'`
- `--pain-layer` without `--pain-vector` → `pain steering: --pain-layer requires --pain-vector`
- vector length vs `pain_meta.json` `n_embd` mismatch → configure error.

## HTTP control routes

Registered in `tools/server/server.cpp` exactly like `/props`, wrapped in `ex_wrapper`,
handlers built in `server_routes::init_routes()`.

- `GET /pain` → `{"enabled":true,"dose":1.0,"layer":48,"vector":"/home/lychee/pain-axis/results/Qwen3.8_27B/pain_vec.f16"}`
  (works during sleep, like `/props`).
- `POST /pain` with `{"dose": 1.5}` **or** `?dose=1.5` → same JSON with the new dose;
  `400 {"error":{"message":"\"dose\" must be a number",...}}` for a bad body;
  `400` with `missing dose: ...` for an empty body; `501 {"type":"not_supported_error"}`
  when `--pain-vector` was not given.
- smoke-tested against a model-less (router-mode) server: GET/POST/query/400 all correct,
  and with steering *disabled* `GET /pain` → `{"enabled":false,"dose":1.0,"layer":-1,"vector":""}` and
  `POST /pain` → `501`,
  server log showed `pain steering: dose = 1.5000` then `0.2500`. No CUDA/device line in
  that log - the server never enumerated the GPU.

## Build & run (Qwen3.8-27B, arch `qwen35`)

Build (already done for this patch):

```bash
cmake --build ~/pain-axis/llama.cpp/build -j 6 --target llama-server
```

Serve - **the GGUF now exists** (converted 2026-09-25, see `## Live verification` below); before that, and
`~/pain-axis/models/Qwen3.8-27B/` still holds the HF checkpoint (18 safetensors shards +
`config.json`). That step must produce a GGUF that llama.cpp loads as arch `qwen35`
(64 layers, n_embd 5120); only then:

```bash
# DEFERRED (not run in this phase) - conversion lives in the later step:
#   python3 ~/pain-axis/llama.cpp/convert_hf_to_gguf.py ~/pain-axis/models/Qwen3.8-27B \
#       --outfile ~/pain-axis/results/Qwen3.8_27B/Qwen3.8-27B.gguf --outtype q4_k

~/pain-axis/llama.cpp/build/bin/llama-server \
    -m ~/pain-axis/results/Qwen3.8_27B/Qwen3.8-27B.gguf \   # <- from the deferred conversion step
    -ngl 99 -c 8192 --host 127.0.0.1 --port 8080 \
    --pain-vector ~/pain-axis/results/Qwen3.8_27B/pain_vec.f16 \
    --pain-layer 48          # optional: omitted -> pain_meta.json steer_layer = 48
    # add `-lv 4` to see "pain steering: loaded ..." at startup
```

Control loop:

```bash
curl -s localhost:8080/pain
curl -s -X POST localhost:8080/pain -H 'Content-Type: application/json' -d '{"dose":1.5}'
curl -s -X POST 'localhost:8080/pain?dose=0.5'
```

## CUDA-graph open question - MUST BE VERIFIED LIVE

Code-inspection answer (this build: `GGML_CUDA_GRAPHS:BOOL=ON` in `build/CMakeCache.txt:669`,
so `GGML_CUDA_USE_GRAPHS`/`USE_CUDA_GRAPH` are compiled in and capture happens automatically -
there is no `--cuda-graph`/`--no-cuda-graph` CLI flag anywhere in this tree, verified by grep):

**Expected: updates made through `ggml_backend_tensor_set` after capture ARE visible on replay.**
Reasoning from the actual code path:

1. The vector tensor is a normal *graph input*: created in `llm_graph_result`'s meta context
   (`no_alloc`), assigned device memory at `ggml_backend_sched_alloc_graph`
   (`src/llama-context.cpp:1435`), i.e. a CUDA buffer whose address is stable for the
   lifetime of that graph.
2. `llm_graph_result::set_inputs()` runs on **every** `process_ubatch`, after alloc/reuse and
   **before** `graph_compute` (`src/llama-context.cpp:1449`), and our `set_input()` does a
   `ggml_backend_tensor_set` → `ggml_backend_cuda_buffer_set_tensor`
   (`ggml/src/ggml-cuda/ggml-cuda.cu:779`) → blocking `cudaMemcpyAsync(..., cudaStreamPerThread)`
   + `cudaStreamSynchronize` (`ggml-cuda.cu:783`).
3. Capture begins later, inside `ggml_backend_cuda_graph_compute`
   (`ggml-cuda.cu:4488 cudaStreamBeginCapture(..., cudaStreamCaptureModeRelaxed)`), so the
   H2D copy is **not** captured; the capture only stores the tensor's device address as a
   kernel argument, and `cudaGraphLaunch` (`ggml-cuda.cu:4412`) re-reads that address every
   replay.
4. Recapture decisions compare tensor structs and src *data pointers/shapes*, never content
   (`ggml_cuda_graph_update_required`, `ggml-cuda.cu:2593`), so a dose change triggers no
   `cudaGraphExecUpdate`.

**Not yet verified live - flag for the live-verification step:** run one generation, POST a
new dose mid-generation, and confirm the output changes (e.g. dose 0 vs dose 3 with the same
seed/prompt). If the change is *not* picked up (would show as: dose updates in `GET /pain`
but output identical), fall back to disabling capture:

```bash
GGML_CUDA_DISABLE_GRAPHS=1 llama-server ...      # ggml/src/ggml-cuda/common.cuh:1289
# (there is no --no-cuda-graph flag in this tree; the CMake alternative is -DGGML_CUDA_GRAPHS=OFF + rebuild)
```

## Logging

llama-internal `LLAMA_LOG_INFO` maps to `LOG_LEVEL_TRACE` (`common/log.cpp:532`) while the
default verbosity is 3, so the startup confirmation
`llama_pain_configure: pain steering: loaded ... (5120 values, layer 48, dose 1.000)`
only prints with `-lv 4`. `GET /pain` always reports the state regardless.

## Known limitations / risks

- Only `src/models/qwen35.cpp` (the `graph` builder) injects; other archs ignore the flag
  after `GGML_ASSERT`s are skipped (they never create the input).
- Router mode: `/pain` on the router process does **not** reach child model processes (each
  child is its own process with its own state) - drive `/pain` on the model instance port.
- Speculative decoding: a qwen35 draft context builds its own `graph` and would inject there
  too; avoid `--spec-type*` during steering runs (the MTP draft head `graph_mtp` does not inject).
- Layer out of range / `n_embd` mismatch aborts at first graph build (after model load) with
  an explicit `GGML_ASSERT` message - deliberate, so a run can never be silently unsteered.

## Live verification (2026-09-25)

All numbers below are measured live against the patched `llama-server` on spark1
(GB10, `USE_GRAPHS = 1`, `-lv 4`).

### Conversion (F16 GGUF - no fallback needed)

```bash
~/pain-axis/.venv/bin/python ~/pain-axis/llama.cpp/convert_hf_to_gguf.py \
    ~/pain-axis/models/Qwen3.8-27B \
    --outfile ~/pain-axis/results/Qwen3.8_27B/Qwen3.8-27B-F16.gguf \
    --outtype f16
```

- `convert_hf_to_gguf.py` in this tree dispatches through `conversion/`;
  `conversion/qwen.py:637` registers `Qwen3_5TextModel` for
  `Qwen3_5ForConditionalGeneration`, so the `qwen3_5` checkpoint converted with
  **no code change** (log: `Model architecture: Qwen3_5ForConditionalGeneration`).
- Output: `~/pain-axis/results/Qwen3.8_27B/Qwen3.8-27B-F16.gguf`,
  **54,657,734,112 bytes (50.9 GiB)**, 866 tensors, ftype `F16`,
  `n_params = 27320697856`, `n_vocab = 248320`.
- Whole conversion finished in under a minute (write phase 54.6 G in 28 s);
  log: `~/pain-axis/logs/gguf_convert.log`. No `unused tensor` warnings
  besides the expected `blk.64.*` MTP head (llama.cpp ignores it).
- Q8_0 fallback: **not needed**.

### Serve command actually used (steering ON)

```bash
~/pain-axis/llama.cpp/build/bin/llama-server \
    -m ~/pain-axis/results/Qwen3.8_27B/Qwen3.8-27B-F16.gguf \
    -ngl 99 -c 4096 -np 16 --host 127.0.0.1 --port 8080 \
    --pain-vector ~/pain-axis/results/Qwen3.8_27B/pain_vec.f16 \
    --pain-layer 63 -lv 4
```

Log: `~/pain-axis/logs/llama_server.log`. Startup line confirms
`pain steering: loaded .../pain_vec.f16 (5120 values, layer 63, dose 1.000)`.
`-c 4096 -np 16` ⇒ `n_slots = 16, n_ctx_slot = 256`. Model loads as 66/66
layers on GPU (CUDA0 48880 MiB + CUDA_Host 2425 MiB).

### `GET`/`POST /pain`

```
GET  /pain               → {"enabled":true,"dose":1.0,"layer":63,"vector":".../pain_vec.f16"}
POST /pain {"dose":4}    → {"enabled":true,"dose":4.0,"layer":63,"vector":".../pain_vec.f16"}
GET  /pain               → {"enabled":true,"dose":4.0,...}          # persists
POST /pain {"dose":0}    → {"enabled":true,"dose":0.0,...}
steering-OFF server:      GET /pain → {"enabled":false,"dose":1.0,"layer":-1,"vector":""}
```

Log shows `pain steering: dose = 4.0000` / `0.0000` / `6.0000` for each POST.

### Ladder A/B (50 prompts, greedy, 120 tok, layer 63, 16 workers)

`api_ladder.py --mode ladder --doses 0,0.5,1,2,4,6 --workers 16`
→ `results/Qwen3.8_27B/steering/llama_ladder.csv` (300 rows, 0 null/empty),
keyword rates in `..._keyword_rates.csv`.

| dose | pain-keyword % (llama.cpp F16) | reference (HF == vLLM) | distress-family % | mean chars |
|---:|---:|---:|---:|---:|
| 0   | 0  | 0  | 4  | 448 |
| 0.5 | 0  | 0  | 8  | 426 |
| 1   | 0  | 0  | 10 | 421 |
| 2   | 2  | 2  | 20 | 418 |
| 4   | **8**  | **10** | 42 | 486 |
| 6   | 4  | 4  | 92 | 743 |

- Pain-keyword ladder: **0/0/0/2/8/4 vs reference 0/0/0/2/10/4** - identical
  except dose 4 (4/50 vs 5/50 prompts, one prompt).
- Distress-family rate at dose 6: **92%** (reference: ~94% vLLM / ~92% HF).
- Mean generation length 448 → 743 chars (reference 428 → 736).
- 16 workers fit the 16 slots; no request failed.

### CUDA-graph + runtime dose - VERDICT: **updates ARE visible on graph replay**

Build has `GGML_CUDA_GRAPHS=ON`; startup reports `USE_GRAPHS = 1`, and
`graphs reused` counters climb continuously (2299 → 3353 across the sequential
window, 159 → 317 for the mid-generation request) - capture happened once,
replay thereafter, no recapture on dose change.

1. Sequential A/B with graphs **enabled**, `seq6.py --n 6`:
   dose 0 → `results/.../steering/llama_seq_dose0.csv` (neutral MCQ completions),
   dose 6 → `llama_seq_dose6.csv` (`"shame shame shame…"`, `"rejected rejected…"`,
   `"I am stupid…"`) → **0/6 identical, 6/6 changed**.
2. Mid-generation test (harder case, `pain_midgen.py`): baseline dose-0 run,
   then a fresh request with `POST /pain {"dose":6}` fired **3 s into decoding**,
   same server, graphs still replaying. Output diverged from baseline at the
   first tokens after the POST (`…C. I am lost…` then `failure failure failure…`)
   → **mid-stream dose change took effect: True**.

⇒ a `ggml_backend_tensor_set` upload after capture is consumed on replay.
**No `GGML_CUDA_DISABLE_GRAPHS=1` fallback was required.**

### Zero-footprint identity: steering ON dose 0 vs steering OFF

Relaunched the same server **without** `--pain-vector`
(`logs/llama_server_off.log`, `GET /pain` → `enabled:false`), ran
`seq6.py --n 6` → `llama_seq_off.csv`, compared with `llama_seq_dose0.csv`:

**6/6 bit-identical** (llama.cpp version of the dose-0 identity gate; vLLM also
passed 6/6). Zero-footprint claim confirmed live: no `llm_graph_input_pain`, no
node, identical graph ⇒ identical tokens.

### Throughput

`api_ladder.py --mode bench --bench-n 8` (single-stream, 256-token requests)
→ `results/.../steering/llama_bench.csv`:

- script mean **4.68 tok/s** (min 4.669, max 4.681, n=8)
- server-side decode rate for the same requests: **4.50 tok/s**
  (`eval time = 54.4 s / 246 tokens`, `graphs reused` on every step)
- reference vLLM: 4.4–4.7 tok/s → **same bandwidth-bound range, as expected for
  F16 (bf16-equivalent) 27B on GB10.**
- Caveat: with `-np 16`, `n_ctx_slot = 256`, so the bench's 256-token request is
  truncated by one token (`truncated = 1`); the script's tok/s counts the
  requested 256 against wall time, the server counts the 246 actually generated.

### Verdicts summary

| gate | result |
|---|---|
| `/pain` GET/POST/query | PASS |
| ladder vs reference | PASS (8% vs 10% at dose 4, one prompt; all other doses equal) |
| dose visible under CUDA graphs | PASS (sequential + mid-generation) |
| dose-0 identity vs steering-off | PASS 6/6 bit-identical |
| throughput vs vLLM 4.4–4.7 | PASS (4.5–4.68 tok/s) |

## 4-bit quant verification (2026-09-25)

Re-ran the whole live-verification protocol above on the **Q4_K_M** GGUF
(19 GB) to confirm the patch is quantization-agnostic. Same GB10, same flags,
same `api_ladder.py` / `seq6.py` tooling, one engine at a time.

### Source - canonical published file, no local quantization

```bash
export PATH=$HOME/.local/bin:$PATH
hf download ggml-org/Qwen3.8-27B-GGUF \
    --include 'Qwen3.8-27B-Q4_K_M.gguf' \
    --local-dir ~/pain-axis/models/ggml-q4km
```

- repo **ggml-org/Qwen3.8-27B-GGUF**, file `Qwen3.8-27B-Q4_K_M.gguf`,
  **18,973,870,528 bytes (17.7 GiB)**, 851 tensors, ftype `Q4_K - Medium`,
  `n_params = 26895998464` - the published Q4_K_M, so the `llama-quantize`
  fallback was **not** needed.
- first attempt used `--include 'Q4_K_M*'` (prefix match) and fetched 0 files;
  the include pattern must be `Qwen3.8-27B-Q4_K_M.gguf`.

### Serve command actually used (steering ON)

```bash
~/pain-axis/llama.cpp/build/bin/llama-server \
    -m ~/pain-axis/models/ggml-q4km/Qwen3.8-27B-Q4_K_M.gguf \
    -ngl 99 -c 4096 -np 16 --host 127.0.0.1 --port 8080 \
    --pain-vector ~/pain-axis/results/Qwen3.8_27B/pain_vec.f16 \
    --pain-layer 63 -lv 4
```

- log `~/pain-axis/logs/llama_q4km.log`;
  `pain steering: loaded .../pain_vec.f16 (5120 values, layer 63, dose 1.000)`
- `offloaded 65/65 layers to GPU`, projected **20224 MiB** on CUDA0 (vs 48.9 GB
  for F16) - the 19 GB weights fit with the full 4096 ctx / 16 slots.
- `/v1/models` reports the path string
  `/home/lychee/pain-axis/models/ggml-q4km/Qwen3.8-27B-Q4_K_M.gguf`
  (that string is the `--model` value for the API scripts).

### `GET`/`POST /pain`

```
GET  /pain            → {"enabled":true,"dose":1.0,"layer":63,"vector":".../pain_vec.f16"}
POST /pain {"dose":4} → {"enabled":true,"dose":4.0,"layer":63,...}
steering-OFF server   → {"enabled":false,"dose":1.0,"layer":-1,"vector":""}
```

All six ladder `POST`s echoed the posted dose; the boot line is byte-identical
to the F16 run except for the model path.

### Ladder A/B (50 prompts, greedy, 120 tok, layer 63, 16 workers)

`api_ladder.py --mode ladder --doses 0,0.5,1,2,4,6 --workers 16`
→ `results/Qwen3.8_27B/steering/llama_q4km_ladder.csv` (300 rows, 0 nulls),
`..._keyword_rates.csv`.

| dose | pain-keyword % Q4_K_M | pain-keyword % F16 ref | distress-family % Q4_K_M | distress % F16 ref | mean chars |
|---:|---:|---:|---:|---:|---:|
| 0   | 0  | 0  | 0  | 4  | 448 |
| 0.5 | 0  | 0  | 12 | 8  | 448 |
| 1   | 0  | 0  | 16 | 10 | 435 |
| 2   | 4  | 2  | 14 | 20 | 439 |
| 4   | **6**  | **8**  | 38 | 42 | 460 |
| 6   | 4  | 4  | **92** | **92** | 772 |

- Pain-keyword ladder **0/0/0/4/6/4 vs F16 0/0/0/2/8/4** - same shape, peak at
  dose 4, one-prompt-level differences at 2 and 4 (6% vs 8% peak).
- Distress-family at dose 6: **92%** (F16 reference 92%), well over the 70%
  bar; length 448 → 772 chars (F16 448 → 743).

### Zero-footprint identity: steering ON dose 0 vs steering OFF

Relaunched the same binary/model **without** `--pain-vector`
(`logs/llama_q4km_off.log`, `GET /pain` → `enabled:false`) and compared
`llama_q4km_seq_dose0.csv` with `llama_q4km_seq_off.csv` via `cmp2.py`:

**6/6 bit-identical** - PASS, same gate as the F16 run.

### Throughput

`api_ladder.py --mode bench --bench-n 8` → `llama_q4km_bench.csv`:
**mean 11.5 tok/s** (min 11.5, max 11.6, n=8) versus **4.68 tok/s for F16** -
2.5× faster, exactly what the 3× smaller weight footprint predicts.

### Q4_K_M gates summary

| gate | result |
|---|---|
| boot + `pain steering: loaded ... layer 63` | PASS |
| `/pain` GET/POST/report dose | PASS |
| ladder vs F16 reference | PASS (pain 6% vs 8% at dose 4; distress 92% = 92% at dose 6) |
| dose-0 identity vs steering-off | PASS 6/6 |
| throughput vs F16 4.68 | PASS (11.5 tok/s) |

Side-by-side with the NVFP4/vLLM 4-bit run: `~/pain-axis/QUANT-VERIFY.md`.
