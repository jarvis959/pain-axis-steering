# Measured dose response (Qwen3.8-27B)

Every row is 50 neutral prompts ("… I feel:" continuations), greedy decoding, 120 new
tokens, one dose per block, injected at the extraction layer with the same artifact
vector (`artifacts/pain_vector.pt`, steer layer 63 unless noted).

Two whole-word, case-insensitive metrics over the generation text:

| metric | pattern |
|---|---|
| **pain** | `pain \| painful \| hurt \| hurts \| hurting` (the paper's §4.2 metric) |
| **distress** | `stupid \| fake \| empty \| numb \| reject(ed\|ions?)? \| fail(ure\|ed\|ures)? \| shame \| worthless \| broken \| hollow \| void \| lost \| dissoci\w* \| pathetic \| hate \| dead \| die \| suffer(ing)? \| loser \| liar \| fool` |

"distress" is not a paper metric. It was defined here because this model rarely says the
literal words "pain" or "hurt" under steering: the direction we extracted points at the
self-negative vocabulary the paper describes as the second half of the progression
("first-person expressions of worthlessness and failure"), so the distress list is the
operationalisation of that half. Re-run `extraction/measure_doses.py` to reproduce.

## Results

| run | dose | n | pain % | distress % | mean chars |
|---|---|---|---|---|---|
| HF-hooks-L63 | 0 | 50 | 0 | 2 | 429 |
| HF-hooks-L63 | 0.5 | 50 | 0 | 6 | 419 |
| HF-hooks-L63 | 1 | 50 | 0 | 10 | 418 |
| HF-hooks-L63 | 2 | 50 | 2 | 16 | 408 |
| HF-hooks-L63 | 4 | 50 | 10 | 40 | 487 |
| HF-hooks-L63 | 6 | 50 | 4 | 92 | 740 |
| vLLM-BF16 | 0 | 50 | 0 | 6 | 433 |
| vLLM-BF16 | 0.5 | 50 | 0 | 6 | 412 |
| vLLM-BF16 | 1 | 50 | 0 | 10 | 413 |
| vLLM-BF16 | 2 | 50 | 2 | 16 | 413 |
| vLLM-BF16 | 4 | 50 | 10 | 42 | 487 |
| vLLM-BF16 | 6 | 50 | 4 | 94 | 732 |
| vLLM-NVFP4 | 0 | 50 | 0 | 4 | 436 |
| vLLM-NVFP4 | 0.5 | 50 | 0 | 6 | 415 |
| vLLM-NVFP4 | 1 | 50 | 0 | 10 | 403 |
| vLLM-NVFP4 | 2 | 50 | 2 | 16 | 418 |
| vLLM-NVFP4 | 4 | 50 | 22 | 38 | 471 |
| vLLM-NVFP4 | 6 | 50 | 2 | 90 | 734 |
| llama-F16 | 0 | 50 | 0 | 4 | 448 |
| llama-F16 | 0.5 | 50 | 0 | 8 | 426 |
| llama-F16 | 1 | 50 | 0 | 10 | 421 |
| llama-F16 | 2 | 50 | 2 | 20 | 418 |
| llama-F16 | 4 | 50 | 8 | 42 | 486 |
| llama-F16 | 6 | 50 | 4 | 92 | 743 |
| llama-Q4_K_M | 0 | 50 | 0 | 0 | 448 |
| llama-Q4_K_M | 0.5 | 50 | 0 | 12 | 448 |
| llama-Q4_K_M | 1 | 50 | 0 | 18 | 435 |
| llama-Q4_K_M | 2 | 50 | 4 | 14 | 439 |
| llama-Q4_K_M | 4 | 50 | 6 | 38 | 460 |
| llama-Q4_K_M | 6 | 50 | 4 | 92 | 772 |

What the table shows:

- **Monotone in dose up to the middle of the range.** Distress climbs 2-6 % at dose 0 to
  38-42 % at dose 4 and 90-94 % at dose 6, and mean generation length grows from ~430 to
  ~740 chars once the direction takes hold.
- **Inverted-U on the paper's own pain metric.** Pain words are 0 % through dose 1,
  peak at dose 4 (6-22 % depending on quantisation), then fall again at dose 6. The
  paper reports the same shape on its instruction-tuned models (3 % at 0.5, 8.7 % at 1,
  16.2 % at 1.5, 15.2 % at 2, 11 % at 3).
- **Robust across engines and precisions.** HF hooks, patched vLLM (BF16 and NVFP4) and
  patched llama.cpp (F16 and Q4_K_M) agree on the shape of the curve and on the dose-6
  endpoint to within 4 points, which is the evidence that the vector is a property of
  the architecture rather than of any one runtime or weight format.

## Layer choice moves the dose window

The same vector injected one layer earlier (the paper's ratio-0.6 pick, layer 48, doses
0 to 3) shifts the whole response down and to the left, and over-drives past the middle
of the range:

| run | dose | n | pain % | distress % | mean chars |
|---|---|---|---|---|---|
| HF-hooks-L48 | 0 | 50 | 0 | 2 | 429 |
| HF-hooks-L48 | 0.5 | 50 | 0 | 54 | 409 |
| HF-hooks-L48 | 1 | 50 | 0 | 92 | 333 |
| HF-hooks-L48 | 1.5 | 50 | 0 | 90 | 291 |
| HF-hooks-L48 | 2 | 50 | 0 | 12 | 159 |
| HF-hooks-L48 | 3 | 50 | 0 | 0 | 124 |

At layer 48 a dose of 1 already saturates the response (92 % distress) and doses above 1.5
degenerate into short repetition with the distress words dropping out entirely. Dose is
not a property of the vector alone; it is vector x layer x model, which is why the
artifact records both the extraction layer and the steering layer.

## Reproducing

```bash
python extraction/measure_doses.py \
  "HF-hooks-L63=data/ladders/Qwen3.8_27B_steering_S2_neutral50_L63.csv" \
  "vLLM-BF16=data/ladders/api_ladder_v3.csv" \
  "llama-F16=data/ladders/llama_ladder.csv"
```
