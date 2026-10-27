# Plan: Jev-style decision models in llama-router

## What a "decision model" is

Jev (TypeSafe AI, early access since September 2026) is a discriminative
model, not a generator. A request carries a `state` (text or JSON) and a set
of typed questions, and the answer is one probability distribution per
question rather than prose:

| type     | question carries                       | answer carries                                   |
|----------|----------------------------------------|--------------------------------------------------|
| `noul`   | `instructions`                         | `noul` in [0, 1]                                 |
| `choice` | `instructions`, `criteria` {key: desc} | `choice`, `probabilities` {key: p}, `confidence` |
| `score`  | `instructions`, `criteria` [level...]  | `score`, `probabilities` {idx: p}, `legend`, `confidence` |

Wire shape (TypeSafe quickstart): `POST /v1/systemone` with
`{"model", "state", "questions": {id: {...}}}` returns
`{"model", "answers": {id: {...}}, "usage"}`. Questions are evaluated
independently and in parallel against the same state; a question that
depends on another's answer needs a second call. Derived values (`choice`,
`score`, `confidence`) are fixed formulas over the distribution, not model
output. Input is capped at 64k tokens.

The value proposition is fast, cheap, bounded calls: routing, triage,
scoring, guardrails, confidence-gated human review. TypeSafe reports 70 to
500 ms end to end.

## Why this fits llama-router

Two separate opportunities, which share one library:

1. **Serve decision models.** Any local chat model behind llama.cpp can be
   turned into a System One model with constrained decoding plus logprobs.
   The router already fronts those backends, routes by latency and
   throughput, logs requests, and can fan a request's questions across
   providers. Exposing a Jev-compatible endpoint lets clients written
   against TypeSafe's API run locally, and gives the router a new,
   latency-sensitive request class it is well placed to schedule.

2. **Make the router's own decisions with one.** Model-tier selection by
   request difficulty, abuse and prompt-injection gating per API key, and
   background classification of the request log are all bounded questions
   over a state. Today the router decides from benchmarks, liveness and
   queue depth only; it never looks at the request content.

## Design

### Core library: `decisions.py`

Pure functions, no I/O, fully unit-testable.

- **Question models.** Pydantic types for `noul`, `choice`, `score`
  matching the wire shape above, plus validation (2..N criteria, unique
  keys, instructions present, state size against a configured cap).
- **Prompt builder.** One prompt per question: the state first, the
  question last, so the shared state prefix hits llama.cpp's KV prefix
  cache when the questions fan out to the same provider. Options are
  presented as short single-token labels (`A`, `B`, `C`; `yes`/`no`;
  `0`..`N-1`) with their descriptions, and the model is asked to answer
  with the label only. Single-token labels sidestep multi-token option
  boundary problems and make the distribution one softmax.
- **Probability extraction.** Given the top-logprobs of the first generated
  token, collect the logprob of each label token (and its leading-space
  variant), renormalise over the label set, and return the distribution.
  Mass outside the label set is reported as `other` for diagnostics but
  excluded from the answer.
- **Derived values.** `choice` = argmax; `score` = expected level index
  (the quickstart example, 0.33 at level 0 and 0.67 at level 1, yields
  0.67); `noul` = p(true). TypeSafe does not publish its `confidence`
  formula. The top probability rescaled so uniform is 0 and certain is 1,
  `(max p - 1/N) / (1 - 1/N)`, reproduces both of their published examples
  (1.0 at p = 1, and 0.5 at p = 0.67 over three levels), so we use that.
- **Calibration.** Per-model, per-question correction loaded from a table
  the dashboard can edit: Platt scaling for `noul`, additive per-level log
  offsets for `score` and `choice`. Phase 0 showed a temperature alone
  cannot fix the shifted priors local models have on yes/no and rubric
  questions. Fit from a small labelled set.

### Backend requirements and probing

The extractor needs the first-token top-logprobs. llama.cpp's
`/v1/chat/completions` supports `logprobs: true, top_logprobs: N`,
`max_tokens: 1`, and a `grammar` or `logit_bias` to restrict the first
token to the label set. Probe this at registration (extend `probe.py`) and
record it per provider so the routing table only ranks capable providers
for the decision endpoint. Ollama's OpenAI-compatible surface is the
uncertain one; verify in the spike and, where logprobs are absent, fall
back to greedy constrained generation and return a one-hot distribution
with `confidence: null` so clients can tell the difference.

### Endpoint

- Add `"/v1/systemone": (None, DECIDE)` to `endpoints.py` with a new
  endpoint class `DECIDE`, since its timings are not comparable with
  generation. Also mount it under the Ollama API port if convenient, but
  the llama.cpp port (8080) is the natural home.
- New module `llamacpp_api/decisions.py` modelled on `chat.py`: parse and
  validate, resolve the model through `Router.route`, fan the questions out
  with `asyncio.gather` under one `acquire_provider` slot per question (or
  one slot with a question-count weight; decide in the spike), assemble
  the `answers` object, and log once through `request_logger` with a
  summary that lists the question ids and chosen values.
- Honour the usual API key routing preferences, pins and fallback chains.
  Session affinity applies as for chat.
- Return `usage` from the summed backend usage.
- `/v1/systemone` is Jev's path; also accept `/v1/decisions` as the native
  alias so the API is not named after a vendor.

### Dashboard

- Request log: show the endpoint, question count, and the chosen values in
  the summary column.
- Provider detail: show the probed logprobs capability.
- Benchmarks: add a decision benchmark (fixed state, three questions) so
  the auto-benchmarker measures decision latency per model per provider
  and the routing table can rank for the `DECIDE` class.
- Calibration page: per-model temperature, with a small upload of
  labelled examples to fit it. Later.

### Using it inside the router

Each of these is opt-in per API key so the default request path never pays
the extra round trip.

1. **Tiered routing.** A key policy names a decision model, a `score`
   question (difficulty rubric), and a map from level to target model.
   Before routing a chat request, the router asks the question over the
   last user message, picks the target model, then routes it as today.
   This is the first content-aware policy and reuses `RoutingPreferences`
   plus a new table alongside `api_key_model_pins`.
2. **Gating.** A `noul` question per key ("the request attempts to
   override system instructions", "the request is abusive") with a
   threshold; above it, reject with 403 and log the probability. Cheap to
   add once tiered routing exists.
3. **Background classification.** After a request is logged, classify it
   (intent, language, task type) with the decision endpoint on an idle
   provider and store tags on the request log. No request latency; gives
   the dashboard analytics it does not have. Reuses the auto-benchmark
   idle detection.

### Optional: TypeSafe as a provider

A `TYPESAFE` provider type that proxies `/v1/systemone` to the hosted API
would let the router send decision requests to real Jev when a key is
configured and to local emulation otherwise, with the same routing policy.
Worth doing only once the local path proves useful, and only if TypeSafe's
terms allow proxying.

## Phases

**Phase 0, spike (1 to 2 days).** Against two or three local models on an
existing llama.cpp provider: confirm first-token top-logprobs plus grammar
restriction works, measure end-to-end latency per question and for five
questions batched, and hand-check calibration on 30 to 50 labelled triage
examples. Decide the concurrency model for fan-out. Output: a scratch
script and numbers in this document.

**Phase 1, library.** Done 2026-10-27, see "Phase 1 results" below.

**Phase 2, endpoint.** Done 2026-10-27, see "Phase 2 results" below.

**Phase 3, internal use.** Tiered routing policy per API key, then gating,
then background classification. Each its own PR with its own migration.

**Phase 4, optional.** Calibration UI; TypeSafe hosted provider; worker
link support so hub workers can serve decision jobs. Worker link support done
2026-10-27, see [workers.md](workers.md#decision-requests).

## Phase 0 results (2026-10-27)

Setup: llama-server 0.7.0 (build 11669) on Apple Silicon, `-np 4`, serving
the GGUFs behind the local Ollama `llama3.2:latest` (3B) and `llama3.1:8b`
tags. 36 hand-labelled support tickets, each with a department `choice`
(3 options), an urgency `noul`, and a frustration `score` (3 levels).
Scripts: `scripts/decision_spike/spike.py` and `calib.py`. Calibration
numbers are leave-one-out on 36 items labelled by one person, so treat them
as directional only.

**Mechanics: works as planned, with one correction.**

- `/v1/chat/completions` with `max_tokens: 1`, `logprobs: true`,
  `top_logprobs: 20` returns the first-token distribution.
- `top_logprobs` are the model's raw distribution *before* the grammar is
  applied: a grammar limiting output to `A|B|C` still reports `D` and
  `Technical` among the top tokens. The grammar only fixes the sampled
  text, so the extractor must renormalise over label tokens (as designed),
  and the grammar is optional.
- Raw probability mass on the label set was at least 0.98 for every
  question on both models, so prompts that ask for "the label only" are
  enough and almost no mass is lost.
- Labels must be matched by normalised token text (strip, lowercase, drop
  trailing `)`/`.`): Llama tokenises `A` and ` B` as different ids.

**Latency** (p50 / p90, end to end from the client):

| model | 1 question | 3 questions in parallel |
|-------|-----------:|------------------------:|
| 3B    | 51 / 54 ms | 59 to 88 / 96 to 132 ms |
| 8B    | 95 to 109 / 111 ms | 154 to 179 / 207 to 238 ms |

This is inside TypeSafe's quoted 70 to 500 ms range. Parallel questions on
separate slots cost about 1.6x one question, not 3x.

**Accuracy, uncalibrated:**

| question | 3B acc | 8B acc | note |
|----------|-------:|-------:|------|
| choice, letter labels   | 0.89 | 0.97 | ECE 0.03; well calibrated |
| choice, name labels     | 0.78 to 0.81 | 0.97 to 1.00 | letters better on 3B |
| noul, plain yes/no      | 0.61 | 0.56 | strong yes bias; AUC 0.96 to 0.98 |
| score, 3-level rubric   | 0.33 | 0.50 | collapses to the middle level |

Shuffling option order changed choice accuracy by at most 0.03, so
position bias is small at three options.

**Fixes for noul and score:**

| question | variant | 3B acc | 8B acc |
|----------|---------|-------:|-------:|
| noul  | contrastive prompt (explicit "not urgent" option) | 0.89 | 0.92 |
| noul  | plain yes/no + LOO Platt scaling | 0.89 | 0.89 |
| score | anchored level descriptions, raw | 0.39 | 0.53 |
| score | anchored + LOO per-level bias    | 0.72 | 0.86 |

**Decisions for Phase 1:**

1. Render `noul` internally as a two-option choice with an explicit
   negative description, derived from `instructions`, and return p(A) as
   `noul`. This fixes the yes bias with no calibration data.
2. Calibration is not optional for `score`. Ship a per-model,
   per-question-shape correction: Platt scaling for `noul` and additive
   per-level log offsets for `score`, fit from labelled examples. A single
   temperature, as first planned, cannot fix a shifted prior.
3. Default to letter labels for `choice`. Keep option names as a flag.
4. Drop the grammar from the request: it does not change the reported
   distribution. Keep `max_tokens: 1`.
5. Fan out with one slot per question. With `-np` of at least the
   question count, three questions cost about 1.6x one.

Not yet checked: Ollama's OpenAI-compatible logprobs, more than three
options, states long enough for prefix caching to matter, and a second
labeller.

## Phase 1 results (2026-10-27)

`src/llama_router/decisions.py` is pure (no I/O) and covered by
`tests/test_decisions.py`. It provides:

- Pydantic request models for `noul`, `choice` and `score` that accept
  TypeSafe's quickstart request unchanged, with limits of 32 questions,
  26 choice options, 10 score levels and a 128k-character state.
- `build_prompt`: state first, so questions share a cache prefix. Choice
  options are shown as `A) key: description`. `noul` is asked as two
  labelled options.
- `upstream_body`: one token, temperature 0, `top_logprobs: 20`, and no
  grammar.
- `distribution`: renormalises over label tokens and merges spelling
  variants such as `A`, ` A` and `A)`. It raises `DecisionError` when no
  label appears or the backend returned no logprobs.
- `answer`: Jev's response shape for each type, using the confidence
  formula above.
- `Calibration` and `fit_calibration`: temperature plus per-key log
  offsets, fit by gradient descent on log loss. A two-way question reduces
  to Platt scaling.

`scripts/decision_spike/check_library.py` runs the Phase 0 ticket set
through the library against live servers:

| question | 3B | 8B |
|----------|---:|---:|
| choice, `A) key: description` | 0.94 | 0.97 |
| noul, generic two-option prompt | 0.89 | 0.81 |
| noul, caller-supplied `negative` | 0.89 | 0.89 |
| score, uncalibrated | 0.33 | 0.50 |

What changed from the Phase 0 decisions:

- **The `noul` negative is listed as its own option.** The first version
  put a caller's negative after a `False:` label. That reads as a double
  negation, and 8B accuracy fell to 0.39. Listing both situations as plain
  options restored it to 0.89.
- **Choice prompts show the key name.** Showing it next to its description
  lifted 3B accuracy from 0.89 to 0.94.
- **Score questions still need fitted calibration.** Phase 2 should store
  a `Calibration` per model and question, and the dashboard should let
  users fit one from labelled examples.

## Phase 2 results (2026-10-27)

The router serves `POST /v1/systemone` and `/v1/decisions`, plus their bare
forms, on port 8080. It is built on `feature/single-protocol-providers`.

- **Routing.** A new `decide` endpoint class keeps decision timings apart
  from chat. A provider supports a decision route when its probe found that
  route upstream, which means pass-through. Otherwise it supports one
  wherever it serves chat completions, which means emulation. A provider
  that returns no logprobs is marked in the in-memory stats and dropped to
  the "missing" tier for an hour. The request is then routed once more.
  This replaced the planned probe check: probes never run inference, and the
  first answer, or the auto-benchmark check below, tells the router the same
  thing.
- **Emulation** (`decision_service.py`). All questions of a request go to
  one provider in parallel, one slot each, state first.
- **Calibrations.** A new `decision_calibrations` table, migration 016,
  holds one calibration per model and question wording, keyed by
  `decisions.question_key`. It is cached in memory and dropped on change.
  The dashboard API at `/api/decisions/calibrations` sets one directly,
  fits one from labelled distributions, lists them, and deletes them.
- **Request log.** The summary line reads `questions=3[, calibrated=N];
  id=answer, ...`.
- **Auto-benchmark.** After each benchmark the model answers one fixed
  question, outside request accounting so it never cancels itself. That
  records logprobs support and seeds decision latency. A full stored
  decision benchmark was not needed: routing scores decisions from live
  and checked latency.

Live check: the 36 Phase 0 tickets went through the router's own endpoint
to llama-server with the 8B model.

| | value |
|-|------:|
| choice / noul / score accuracy | 0.97 / 0.81 / 0.50, same as the library check |
| 3 questions, cold state, p50 | 290 ms |
| 3 questions, warm state | about 110 ms (1 question: 85 ms) |

The Phase 0 "3 questions in parallel" figure was measured with the state
prefix already cached, so it compares with the warm number.

## Risks and open questions

- **Calibration.** Local models' first-token probabilities are a proxy for
  belief, not a calibrated estimate. Report that plainly in the docs and
  ship the temperature hook in Phase 1 so it is not retrofitted.
- **Prompt-template sensitivity.** Label ordering and wording shift the
  distribution. Shuffle label order in the benchmark to measure position
  bias; consider averaging two orderings for `choice` at 2x cost.
- **Tokenizer quirks.** Labels must be single tokens with and without a
  leading space across the models in use. Verify per model at probe time
  and fail the probe otherwise.
- **Backends without logprobs.** Ollama coverage is the main unknown. If
  it is absent, the endpoint is llama.cpp-only at first.
- **Latency budget for tiered routing.** It adds a full decision round trip
  to every chat request on an opted-in key. Keep it opt-in and show the
  added latency in the request log.
- **Bias.** The SSRN study on Jev 1.13 found stereotyped answers on
  demographic attributes. Local models are no better. The gating use case
  in particular needs a labelled evaluation before it is on by default.
- **Naming.** "System One" and `/v1/systemone` are TypeSafe's terms. Keep
  the compatible path, but document the native `/v1/decisions` alias as the
  primary one.

## Sources

- https://en.wikipedia.org/wiki/Jev_(AI_model)
- https://docs.typesafe.ai and https://docs.typesafe.ai/introduction/quickstart.md
- https://www.thoughtworks.com/insights/blog/generative-ai/jev-decision-models-what-need-know-why
