# Cross-repository issue work doc

Prioritize safe inference, additional model families and usable model formats. Child-repository work moves up when it enables those outcomes. Small optimizations and peripheral platforms stay at the bottom.

Audit snapshot: **15 October 2026 · 148 open GitHub issues · four repositories**. Tiers are planning recommendations, not changes to GitHub labels. Every inventory row starts open; the initial audit does not change GitHub.[^scope] Status notes include later completed work; counts retain the audit snapshot.

## Priority rules

| Tier | Meaning | Selection rule |
|---|---|---|
| P0 | Main-path correctness/blockers | Known SBCL-path correctness defects, including defects currently contained by a fallback |
| P1 | Model and inference capability | Model families, formats, session features and their necessary child-repo prerequisites |
| P2 | Substantial performance and reliability | Measured material bottlenecks, CI, supporting APIs and ECL/CCL correctness or validation |
| P3 | Deferred | Small performance improvements, convenience, speculative ceilings, other Lisps, Windows/BSD and peripheral targets |

SBCL is primary. ECL and CCL are secondary. Windows/BSD remain P3 even when the Lisp is ECL or CCL. A broad child-library feature does not earn P1 simply because it adds an API. P1 requires a concrete inference capability or prerequisite. GPU expansion ranks above generic BLAS completeness; f64 emulation is not a prerequisite for an f32/f16-first Metal tensor backend.

## Counts

| Repository | P0 | P1 | P2 | P3 | Total |
|---|---:|---:|---:|---:|---:|
| cl-inference | 1 | 10 | 5 | 8 | 24 |
| cl-tensor | 0 | 1 | 8 | 9 | 18 |
| cl-gpu | 0 | 7 | 13 | 23 | 43 |
| trivial-simd | 1 | 0 | 10 | 52 | 63 |
| **Total** | **2** | **18** | **36** | **92** | **148** |

## Work order

1. **Protect existing execution.** Retain message control-token protection and its adversarial fixtures. Native selection and tensor fusion retain lazy-branch regression coverage. Run existing fixture suites before adding functionality.
2. **Deliver one model family end to end.** Choose an initial Gemma variant, then implement the tokenizer, rule ordering and template behavior it actually requires. Pin a model and reference for tokenization, logits and greedy generation. Retain default GGUF tokenizer regression coverage when adding families.
3. **Expand useful inference.** Add multi-turn cache reuse, then evaluate quantized KV storage for longer contexts. Keep prefix-cache ownership, reset and cancellation in the acceptance scope.
4. **Choose the next execution expansion.** ONNX requires an import/runtime decision. For GPU work, choose direct tensor storage or MLX format expansion according to the models being targeted; do not implement both as a prerequisite for either.
5. **Remove measured bottlenecks.** Retain the caller-owned tensor executor; measure mapped F32 copies, Q8 prefill, irregular half-weight packing and MLX cache traffic on real models before smaller loop tuning.

These are ordered workstreams, not estimates or commitments to complete every P1 ticket before P2. CI and conformance work should accompany each workstream. User approval is required for model variants, new runtime dependencies and API design; this audit does not settle those designs.

## Dependency paths

| Outcome | Work path | Boundary |
|---|---|---|
| Gemma acceptance | Select family/version → architecture and weights → required tokenizer/template → client/session acceptance | SentencePiece and family-specific behavior depend on the chosen variant |
| More MLX model artifacts | Select quantized formats → additional affine widths, or embeddable cl-gpu output → MLX custom kernels | Custom kernels are useful for unsupported GGUF quant formats; they do not require cl-tensor device storage |
| Direct GPU tensor inference | Axis reductions/normalization + broadcast/conversion → tensor device storage → model integration and acceptance | Existing transfers, array views and matrix primitives are groundwork; f64 can remain explicitly unsupported on Metal |
| NVIDIA/OpenCL execution | Backend runtime/printer + capability/alignment checks → backend-neutral conformance → tensor integration | CUDA vector helpers and OpenCL sub-buffer alignment belong with their backends |
| CPU parallel inference | Existing tensor executor hook → inference integration → measured scheduling thresholds | Single-contraction splitting is separate optional work |
| Lower scratch/validation overhead | Shared upstream proofs/adapters → tensor consumers → complete-call allocation checks | Avoid implementing separate uniqueness solvers at each layer |

## Keeping this doc current

For each selected row, record an owner and a short next action in its planning note. Replace `Open` with `Active`, `Blocked: prerequisite`, or `Closed: date` as work progresses; keep GitHub authoritative. A P1 completion includes fixtures, a small example, current support docs and pinned real-model acceptance where applicable. Close a ticket only after its acceptance criteria are met. Recheck counts and links after each batch; remove closed rows from the active inventory at the next audit.

## Limitations

The inventory does not certify runtime correctness or performance. Validation results apply to selected work and are linked from its status notes. Broad umbrella issues and overlapping follow-ups are counted once per open GitHub issue, not as independent deliverables. Dependencies not explicitly established by ticket text or checked documentation are planning inferences.

### Tracker reconciliation

Follow-up outside the snapshot; the original inventory counts remain unchanged. Live count check after this batch on **16 October 2026**: cl-inference **22**, cl-tensor **18**, cl-gpu **43**, trivial-simd **62** — **145 open issues** total:

| Ticket | State | Planning note |
|---|---|---|
| [cl-inference #53 — HF Qwen empty-string encoding](https://github.com/communal-software/cl-inference/issues/53) | Closed: 2026-10-15 | Owner: Codex. Empty HF Qwen text bypasses NFC normalization and follows the shared BOS policy. HF Qwen/Llama fixtures cover configured and missing BOS, defaults and overrides; combining-character and chat-provenance checks pass. SBCL: 9,584 local checks and 527 client checks pass; three optional live client checks skip. Pinned Qwen empty-input checks pass. See [tokenizer behavior](tokenizers.md#functions). Next: select remaining tokenizer coverage from the inventory. |
| [cl-inference #54 — ECL half-linear profile call](https://github.com/communal-software/cl-inference/issues/54) | Closed: 2026-10-15 | Owner: Codex. The failure reproduces with unchanged compiled baseline code; interpreted source passes. A `notinline` declaration preserves fixture replacement; the isolated fixture and all three local suites pass. Next: preserve cross-Lisp regression coverage in CI. |
| [cl-inference #55 — Parallel half-weight dispatch](https://github.com/communal-software/cl-inference/issues/55) | Closed: 2026-10-16 | Owner: Codex. Encoded single-row F16/BF16 dispatch preserves the caller-owned executor and directly readable mapped/padded rows on M1/SBCL/native. Cross-Lisp fixtures, serial/parallel model acceptance and 5,256 profile rows pass. See [implementation review](#half-weight-dispatch). Next: measure the serial irregular-layout policy tracked below. |
| [cl-inference #56 — Ambiguous function-wrapper scalar values](https://github.com/communal-software/cl-inference/issues/56) | Open | Owner: unassigned. P2 follow-up for untyped/nullable scalar type loss and unescaped delimiters in the pinned format. Ordinary typed values and untyped text/JSON are supported. The parser TODO names explicit codecs as the upgrade path. Next: choose a per-tool codec or explicit ambiguity policy and add collision round trips. |
| [cl-inference #57 — Q8 cache conversion and attention scratch](https://github.com/communal-software/cl-inference/issues/57) | Open | Owner: unassigned. Follow-up for scalar CPU conversion and full-prefix F32 attention scratch; source TODOs name bulk conversion and tiled/fused attention as upgrade paths. Next: use the [cache profile](cache-acceptance.md#recorded-profile) to select the measured bottleneck. |
| [cl-inference #58 — Stale template limitation references](https://github.com/communal-software/cl-inference/issues/58) | Open | Owner: unassigned. Existing template limits refer to closed implementation work. Next: reconcile links while preserving the pinned coverage and explicit semantic limits. |
| [cl-inference #59 — Serial half-weight layout packing](https://github.com/communal-software/cl-inference/issues/59) | Open | Owner: unassigned. Issue 55 profiling exposes a pre-existing serial automatic-dispatch slowdown for stepped/reversed weights: encoded layout packing costs more than tiled conversion. Next: measure a layout-aware serial policy; preserve compatible mapped rows and forced-path correctness. |

| Finding | Evidence and next action |
|---|---|
| Parallel inference integration is complete | [cl-inference #17](https://github.com/communal-software/cl-inference/issues/17) and its [threading prerequisite](https://github.com/communal-software/cl-tensor/issues/2) are closed. Caller-owned head/row scheduling follows the [executor contract](parallel-inference.md). The [half-weight profile](parallel-inference.md#half-weight-dispatch-profile) covers single-row encoded dispatch and its layout boundary. |
| Pinned template-family coverage is complete | [cl-inference #38](https://github.com/communal-software/cl-inference/issues/38) is closed with native family renderers/parsers and strict supplied-source handling. Unlisted variants and raw scalar/delimiter ambiguity remain explicit [limits](chat-families.md#limitations). |
| Tensor GPU prerequisites mix completed groundwork and open scope | [cl-tensor #12](https://github.com/communal-software/cl-tensor/issues/12) comments confirm bulk transfers and array views exist. [cl-gpu #58](https://github.com/communal-software/cl-gpu/issues/58) and [#59](https://github.com/communal-software/cl-gpu/issues/59) remain open. Software f64 is optional for an explicitly restricted dtype backend. |
| MLX custom-kernel issues carry an old low-priority recommendation | [cl-inference #44](https://github.com/communal-software/cl-inference/issues/44) and [cl-gpu #20](https://github.com/communal-software/cl-gpu/issues/20) say low priority. P1 here is conditional on using them for unsupported model formats; keep experimental sampling/fusion tuning P3. |
| Tensor uniqueness work overlaps upstream | [cl-tensor #5](https://github.com/communal-software/cl-tensor/issues/5) and [trivial-simd #63](https://github.com/communal-software/trivial-simd/issues/63) cover related proof workspace. Coordinate upstream before downstream cleanup. |
| Mapped F32 snapshot scope needs reconfirmation | [cl-tensor #6](https://github.com/communal-software/cl-tensor/issues/6) records large projection-copy costs, but mentions an older full-model acceptance. Reproduce on current code before choosing the fix. |
| One open BLAS task appears partly superseded | [trivial-simd #40](https://github.com/communal-software/trivial-simd/issues/40) requests a fused solve kernel; [#41](https://github.com/communal-software/trivial-simd/issues/41) and [#42](https://github.com/communal-software/trivial-simd/issues/42) describe that kernel as present. Verify benchmarks and acceptance before narrowing or closing; do not implement it again from the title. |
| GPU matrix limitation has a known stale link | [cl-gpu #71](https://github.com/communal-software/cl-gpu/issues/71) already tracks the f32-only matrix limitation pointing to closed work. Fix the docs reference when maintaining that repo. |
| Roadmap mixes complete and remaining work | [cl-inference #1](https://github.com/communal-software/cl-inference/issues/1) includes implemented children and a later system-layout clarification. Treat it as coordination, not extra feature work. |

### Selected implementation review

Review date: **15 October 2026**. Owner: Codex. Both selected issues are closed after implementation, correctness acceptance and performance measurements. [Implementation `d99c2c4`](https://github.com/communal-software/cl-inference/commit/d99c2c4) is published to origin and mirror. Snapshot counts remain historical.[^selected-review]

| Validation | Passing checks | Skips |
|---|---:|---:|
| Local fixtures: SBCL, CCL and ECL | 13,714 each | 0 |
| SBCL client | 527 | 3 optional live checks |
| Runnable examples | 9 | 0 |
| MLX fixtures / strict four-model acceptance | 198 / 164 | 0 |
| Pinned CPU acceptance: serial / two workers / four workers | 1,499 / 535 / 535 | 0 |

#### Parallel inference

[#17 — Parallel matmul and attention](https://github.com/communal-software/cl-inference/issues/17) uses caller-owned `ct:*executor*` for tensor rows and independent attention heads. Head results retain order, callbacks inherit backend/FMA bindings, and nested tensor scheduling is disabled. NIL preserves serial execution. Cache writes precede head work; the executor joins callbacks before returning or propagating an error.

| Confirmed decision | Implementation boundary |
|---|---|
| Caller-owned synchronous executor | No session-owned pool or worker-count API |
| Independent attention heads | Ordered result slots and per-callback scratch |
| Existing matmul row scheduling | Output-feature and single-contraction splitting remain separate work |

Pinned CPU acceptance passes for serial, two- and four-worker execution, including serial/parallel logits and greedy output. Fixtures cover floating and packed weights, prefill/decode, bindings, range validation, worker errors, cancellation and deadlines. The [executor example and measurement commands](parallel-inference.md) describe ownership and scheduling policy. The [recorded profile](parallel-inference.md#measured-profile) includes 180 single-native-thread rows and 108 native-default synthetic rows. In that scheduling baseline, longer F16 prefills benefit, tiled F16 decode regresses and Q8 throughput remains near serial. The [half-weight profile](parallel-inference.md#half-weight-dispatch-profile) measures encoded single-row dispatch separately.

#### Half-weight dispatch

Review date: **16 October 2026**. Owner: Codex. The issue is closed after implementation, correctness acceptance and complete-call measurements. [Implementation `f46a0ed`](https://github.com/communal-software/cl-inference/commit/f46a0ed9c527af50f46d168d56e59f7c404654e9) is published to origin and mirror.

The executor-bound automatic profile selects encoded F16/BF16 matmul for one activation row with directly readable weight rows on M1/SBCL/native. Offsets and padded rows qualify; irregular layouts and larger batches use tiles. Serial cutoffs and forced-path behavior retain their existing contract. Single-row execution validates the caller-owned executor without scheduling callbacks.

| Validation | Passing checks | Skips |
|---|---:|---:|
| Local fixtures: SBCL, CCL and ECL | 14,109 each | 0 |
| SBCL client | 527 | 3 optional live checks |
| Runnable examples | 12 | 0 |
| Full pinned CPU acceptance, serial | 1,603 | 0 |
| Focused F16/Q8 acceptance: serial, two workers, four workers | 535 each | 0 |

Existing local suites pass 14,011 checks on each Lisp before implementation. Final fixtures cover mapped offsets, padding, irregular activation/weight views, dispatch fallbacks, unchanged serial cutoffs, executor validation and absence of nested callbacks. Pinned model checks retain relative-logit bounds and exact greedy token IDs.

The [complete-call profile](parallel-inference.md#half-weight-dispatch-profile) retains 5,256 rows: three 936-row single-native-thread model/projection processes and three 816-row native-default projection processes. All 64 eligible single-row mapped/padded groups save at least 10% against tiled calls (3.58–5.67×). With one native thread, executor-bound F16 decode improves 3.26–3.41× and four-token generation improves 1.36–2.26× against baseline dispatch with the same pool. Llama F16 decode Lisp allocation falls from 344.4 to 11.9 MiB in the 16-token/two-worker case.

Model performance evidence covers one native thread; native-default evidence covers projections. Incomplete attempts with macOS background CPU load are excluded. The approved competing jobs resume after profiling. Recorded runtime/build hashes match after measurement; the SIMD checkout contains a documented pre-existing local source snapshot. The serial irregular-layout packing cost is tracked in reconciliation. Live counts verified after closure: cl-inference **23**, cl-tensor **18**, cl-gpu **43**, trivial-simd **62** — **146 open issues**. Historical snapshot counts remain unchanged.

#### Template families

[#38 — Chat templates: more families and stricter detection](https://github.com/communal-software/cl-inference/issues/38) uses native family classes and the existing exclusion mechanism. Supplied unknown source signals `unknown-chat-template`; architecture fallback applies only when source is absent. Caller text retains segment provenance.

The [family matrix](chat-families.md) covers Qwen2.5, Qwen3-Coder, Qwen3.5, QwQ, both R1 distills, Llama 3.0/3.1 and Hermes 2 Pro/3. Llama 3.1 builtin configuration uses declared tool names through `:builtin-tools` and `:tools-in-user-message`. Matching parsers emit existing validated client events without executing code. Native Lisp has no runtime Jinja dependency.

The pinned 12-template matrix passes 477 prompt/error parity checks and 24 detection checks. Streaming boundary, malformed-call, typed-schema, client round-trip, BOS/stop and adversarial control-token checks pass. Existing Llama 3.2/Qwen3 prompt IDs and CPU acceptance pass; strict four-model MLX acceptance passes all 164 checks. Template coverage is independent of architecture support; pinned reference quirks remain explicit [limitations](chat-families.md#limitations).

### Default GGUF tokenizer review

Review date: **16 October 2026**. Owner: Codex. [cl-inference #32](https://github.com/communal-software/cl-inference/issues/32) is closed. [Implementation `2aa7b39`](https://github.com/communal-software/cl-inference/commit/2aa7b395e8f78db478868b4a3c9790bf4acb2690) is published to origin and mirror.

Missing, empty and explicit `default` GGUF pre-types select `default-bpe` for model `gpt2`. Its [four-stage pipeline](pre-tokenizer-stages.md#default-gguf-pipeline) matches the pinned llama.cpp default; unknown names and unsupported models retain errors. BOS metadata and caller overrides retain their contract. Named stages replace whole inherited stages in place, append new stages and preserve earlier piece boundaries.[^default-tokenizer]

| Validation | Passing checks | Skips |
|---|---:|---:|
| Local fixtures: SBCL, CCL and ECL | 15,544 each | 0 |
| SBCL client | 527 | 3 optional live checks |
| Runnable examples | 12 | 0 |
| Full pinned CPU acceptance | 1,603 | 0 |
| Reference regeneration | 150 split cases / 450 tokenizer cases | 0 |

The unchanged SBCL baseline passes 14,109 checks before implementation. Final fixtures cover all three selectors, stage inheritance and redefinition, boundary isolation, unmatched gaps, invalid declarations/options/matches, BOS, special tokens, malformed vocabularies and closed-weight ownership. Existing Llama/Qwen GGUF and HF coverage passes; CPU acceptance includes F16, Q8, Q4/Q6 and Q8 cache references. The documented staged-rule example passes.

Live counts verified after closure: cl-inference **22**, cl-tensor **18**, cl-gpu **43**, trivial-simd **62** — **145 open issues**. Historical snapshot counts remain unchanged. Next: select the first family for broader tokenizer support.

### Cache implementation review

Review date: **15 October 2026**. Owner: Codex. Both cache issues are closed. [Implementation `5a371a5`](https://github.com/communal-software/cl-inference/commit/5a371a57ab5d4609c5b27f5ef282eadc22d2652d) and the [cl-mlx dependency `e9d7d40`](https://github.com/communal-software/cl-mlx/commit/e9d7d40) are published to origin and mirror.

| Validation | Passing checks | Skips |
|---|---:|---:|
| Local fixtures: SBCL, CCL and ECL | 14,011 each | 0 |
| SBCL client | 527 | 3 optional live checks |
| Runnable examples | 12 | 0 |
| cl-mlx / MLX fixtures | 51 / 281 | 0 |
| Pinned CPU acceptance | 1,603 | 0 |
| Strict four-model MLX/cache acceptance | 376 | 0 |

[Prefix reuse](kv-cache.md#prefix-matching) is session-local and enabled by default, with an opt-out. Complete prompts match rendered token IDs; token-ID input still appends unless marked as a complete prompt. Complete hits replay the final token. Serialized backends retain one successful session; failures discard it. Fixtures cover branching, tools/template changes, generated tokens, cancellation, partial-layer failures, threaded attention and MLX ownership.

Q8 is opt-in on CPU and MLX; F32 remains the default. The shared encoding uses signed-byte groups of 32 and F32 scales, including partial groups, ties-to-even and finite-extrema reconstruction. Attention uses temporary F32 prefixes. The user confirms retaining this storage-first format with measured F32 quality bounds of relative L2 ≤0.15 and cosine ≥0.99; independent Q8 oracle comparisons use tighter bounds. All six cases match eight oracle greedy IDs, and those reference sequences match their corresponding F32 references.

The [recorded profile](cache-acceptance.md#recorded-profile) retains **63 MiB Q8 versus 224 MiB F32** at context capacity 1024. Eight-token generation takes about **5.5× longer on CPU and 1.4× on MLX** in this profile. Conversion and scratch tuning remain open; existing full-capacity MLX writes remain a separate limitation. The unrelated stale template-limit links are recorded in tracker reconciliation.

Live counts verified after closure: cl-inference **23**, cl-tensor **18**, cl-gpu **43**, trivial-simd **62** — **146 open issues**. Historical snapshot counts remain unchanged.

### P0 — main-path correctness

| Repository | Ticket | State | Planning note |
|---|---|---|---|
| cl-inference | [#37 — Chat templates: encode message text without parsing special tokens](https://github.com/communal-software/cl-inference/issues/37) | Closed: 2026-10-15 | Owner: Codex. Text-first rendering with segment provenance protects messages, tools, reasoning and dates without changing ordinary prompt IDs. Client/local fixtures, pinned CPU acceptance and strict four-model MLX acceptance pass; only three optional live client tests skip. See [implementation and validation](chat-templates.md#message-data). The separate HF Qwen empty-input fix is recorded under tracker reconciliation. |
| trivial-simd | [#81 — Preserve lazy branches in native elementwise selection kernels](https://github.com/communal-software/trivial-simd/issues/81) | Closed: 2026-10-15 | Owner: Codex. Native selection on `trunk` preserves lazy branches; tensor fusion uses the selected backend. SBCL/CCL/ECL suites pass.[^selection81] Next: tune measured masking overhead. |

### P1 — model and inference capability

| Repository | Ticket | State | Planning note |
|---|---|---|---|
| cl-inference | [#16 — Multi-turn KV reuse and prefix caching](https://github.com/communal-software/cl-inference/issues/16) | Closed: 2026-10-15 | Owner: Codex. Session-local exact-token matching, final-token replay and serialized-backend reuse; opt-out retains fresh prefill. CPU/MLX fixtures, pinned acceptance and the multi-turn example pass. Published to origin and mirror; see [cache review](#cache-implementation-review). Next: preserve prefix regressions when extending model support. |
| cl-inference | [#18 — Quantized KV cache](https://github.com/communal-software/cl-inference/issues/18) | Closed: 2026-10-15 | Owner: Codex. Opt-in symmetric S8/F32 group-32 caches on CPU and MLX, with finite reconstruction and temporary F32 attention prefixes. Pinned quality/oracle acceptance and storage/timing measurements pass; see [cache review](#cache-implementation-review). Next: conversion and scratch tuning under the follow-up in tracker reconciliation. |
| cl-inference | [#32 — Tokenizer: default pre-tokenizer for GGUF files without tokenizer.ggml.pre](https://github.com/communal-software/cl-inference/issues/32) | Closed: 2026-10-16 | Owner: Codex. Missing/empty/default GGUF `gpt2` pre-types select the pinned four-stage default; public named stages replace whole inherited stages in place. Cross-Lisp fixtures, offline reference cases and pinned CPU acceptance pass. Published to origin and mirror; see [tokenizer review](#default-gguf-tokenizer-review). Next: choose the first additional tokenizer/model family under the following row. |
| cl-inference | [#33 — Tokenizer: more pre types and SentencePiece models](https://github.com/communal-software/cl-inference/issues/33) | Open | Tokenizer algorithms and pre-types unlock additional model families; select the first family. |
| cl-inference | [#34 — Tokenizer: insert a child's pre-tokenizer rule before an inherited rule](https://github.com/communal-software/cl-inference/issues/34) | Open | Enables tokenizer inheritance where rule order matters; do alongside a model that needs it. |
| cl-inference | [#38 — Chat templates: more families and stricter detection](https://github.com/communal-software/cl-inference/issues/38) | Closed: 2026-10-15 | Owner: Codex. Native family renderers, matching parsers and strict source fallback pass the pinned reference matrix, provenance/client fixtures and CPU/MLX acceptance. See [implementation review](#template-families). Next: choose explicit codecs for the remaining format ambiguity tracked in the reconciliation table. |
| cl-inference | [#44 — Custom GPU kernels for the MLX backend via cl-gpu](https://github.com/communal-software/cl-inference/issues/44) | Open | Enables GGUF quant kernels in MLX; requires cl-gpu embeddable output. Conditional on format scope. |
| cl-inference | [#47 — Local ONNX model support](https://github.com/communal-software/cl-inference/issues/47) | Open | Adds a model/export execution route; runtime/import approach requires user design approval. |
| cl-inference | [#48 — Local Gemma-family support](https://github.com/communal-software/cl-inference/issues/48) | Open | Adds an architecture family; choose one Gemma version/variant and reference acceptance. |
| cl-inference | [#50 — MLX: additional affine quantization widths and modes](https://github.com/communal-software/cl-inference/issues/50) | Open | Accepts additional MLX quantized models; implement widths required by selected artifacts. |
| cl-tensor | [#12 — Device tensor storage backed by cl-gpu](https://github.com/communal-software/cl-tensor/issues/12) | Open | Enables direct tensor GPU execution; requires axis/math/conversion kernels, not MLX integration. |
| cl-gpu | [#6 — CUDA backend](https://github.com/communal-software/cl-gpu/issues/6) | Open | Adds GPU execution hardware coverage; require backend conformance and an inference consumer. |
| cl-gpu | [#7 — OpenCL backend](https://github.com/communal-software/cl-gpu/issues/7) | Open | Adds GPU execution hardware coverage; require backend conformance and an inference consumer. |
| cl-gpu | [#12 — CUDA: operators for vector types](https://github.com/communal-software/cl-gpu/issues/12) | Open | CUDA vector printer correctness is a backend prerequisite; deliver with CUDA. |
| cl-gpu | [#20 — Embeddable kernel output: body-only printing for hosts that write the signature](https://github.com/communal-software/cl-gpu/issues/20) | Open | Prerequisite for MLX custom kernels; inference model coverage overrides the old low-priority label. |
| cl-gpu | [#58 — Axis reductions, softmax and rmsnorm kernels](https://github.com/communal-software/cl-gpu/issues/58) | Open | Required math, normalization, broadcast and conversion for direct GPU tensor inference. |
| cl-gpu | [#59 — Broadcast elementwise and convert kernels](https://github.com/communal-software/cl-gpu/issues/59) | Open | Required math, normalization, broadcast and conversion for direct GPU tensor inference. |
| cl-gpu | [#73 — OpenCL: sub-buffer base-address alignment for array views](https://github.com/communal-software/cl-gpu/issues/73) | Open | OpenCL view alignment is correctness groundwork; deliver with OpenCL array views. |

### P2 — substantial performance and reliability

| Repository | Ticket | State | Planning note |
|---|---|---|---|
| cl-inference | [#1 — cl-inference roadmap: general-purpose inference on cl-tensor](https://github.com/communal-software/cl-inference/issues/1) | Open | Coordination umbrella, not a separate implementation; reconcile its completed children. |
| cl-inference | [#17 — Parallel matmul and attention](https://github.com/communal-software/cl-inference/issues/17) | Closed: 2026-10-15 | Owner: Codex. Caller-owned execution schedules tensor rows and attention heads; serial/parallel correctness and measured limits are recorded. See [implementation review](#parallel-inference). Next: measure remaining contraction scheduling and serial irregular-layout costs. |
| cl-inference | [#19 — CI](https://github.com/communal-software/cl-inference/issues/19) | Open | Automates fixture checks; keep real-model acceptance opt-in. |
| cl-inference | [#41 — Q8_0 prefill: dequantize tiles and use BLAS for many activation rows](https://github.com/communal-software/cl-inference/issues/41) | Open | Measured long-prompt prefill bottleneck; preserve the fast single-row path. |
| cl-inference | [#49 — MLX KV cache: avoid full-capacity slice updates](https://github.com/communal-software/cl-inference/issues/49) | Open | Context-sized cache traffic can limit long-context decode; measure donation/chunking. |
| cl-tensor | [#4 — CI](https://github.com/communal-software/cl-tensor/issues/4) | Open | Protects tensor correctness on the main development path. |
| cl-tensor | [#6 — Avoid unnecessary snapshots for mixed Lisp and foreign storage](https://github.com/communal-software/cl-tensor/issues/6) | Open | Mapped F32 model projections have a documented large snapshot penalty; verify remaining scope. |
| cl-tensor | [#9 — Investigate incomplete or crashing normalization reference checks on experimental ARM64 CCL](https://github.com/communal-software/cl-tensor/issues/9) | Open | CCL secondary support: green tests can omit assertions; require complete trial counts. |
| cl-tensor | [#14 — Execute fused expressions through upstream strided N-D kernels](https://github.com/communal-software/cl-tensor/issues/14) | Open | Measured reversed-layout fusion regression; upstream N-D expression execution is needed. |
| cl-tensor | [#15 — Support fused mean and widened reduction accumulation](https://github.com/communal-software/cl-tensor/issues/15) | Open | Expands generic fusion expressiveness; promote only when selected models need it. |
| cl-tensor | [#17 — Support fused mask outputs and mask reductions](https://github.com/communal-software/cl-tensor/issues/17) | Open | Expands generic fusion expressiveness; promote only when selected models need it. |
| cl-tensor | [#18 — Parallelize single dot products and whole-tensor reductions](https://github.com/communal-software/cl-tensor/issues/18) | Open | Single contractions remain sequential; benchmark decode-sized workloads before splitting. |
| cl-tensor | [#19 — Support per-reducer axis sets in fused expressions](https://github.com/communal-software/cl-tensor/issues/19) | Open | Expands generic fusion expressiveness; promote only when selected models need it. |
| cl-gpu | [#18 — Kernel pipeline cache keeps pipelines of released devices](https://github.com/communal-software/cl-gpu/issues/18) | Open | Released-device pipeline entries retain resources; fix lifecycle and test reopen/release. |
| cl-gpu | [#22 — Writable swizzles](https://github.com/communal-software/cl-gpu/issues/22) | Open | DSL expressiveness; elevate only if an inference kernel requires it. |
| cl-gpu | [#23 — Vector comparisons and scalar broadcast in intrinsics](https://github.com/communal-software/cl-gpu/issues/23) | Open | DSL expressiveness; elevate only if an inference kernel requires it. |
| cl-gpu | [#32 — CI matrix across SBCL, ECL and CCL](https://github.com/communal-software/cl-gpu/issues/32) | Open | Automates SBCL and secondary ECL/CCL validation. |
| cl-gpu | [#44 — GPU BLAS: register-blocked gemm](https://github.com/communal-software/cl-gpu/issues/44) | Open | Core GEMM/GEMV throughput; measure DSL paths separately from Metal MPS overrides. |
| cl-gpu | [#45 — GPU BLAS: gemv with one group per row](https://github.com/communal-software/cl-gpu/issues/45) | Open | Core GEMM/GEMV throughput; measure DSL paths separately from Metal MPS overrides. |
| cl-gpu | [#47 — GPU BLAS: MPS for broadcast and negative-stride batched gemm](https://github.com/communal-software/cl-gpu/issues/47) | Open | Broadcast batches are relevant to tensor matmul; compare queued MPS against DSL. |
| cl-gpu | [#49 — Metal: host access synchronizes the whole device, not only the buffers in use](https://github.com/communal-software/cl-gpu/issues/49) | Open | Whole-device waits serialize work/transfers; meaningful once device tensors exercise them. |
| cl-gpu | [#51 — Metal: ring of staging buffers so private uploads overlap pending work](https://github.com/communal-software/cl-gpu/issues/51) | Open | Whole-device waits serialize work/transfers; meaningful once device tensors exercise them. |
| cl-gpu | [#52 — Metal: test private buffer storage on a discrete GPU](https://github.com/communal-software/cl-gpu/issues/52) | Open | Validates private-storage correctness on discrete Metal GPUs; hardware-dependent. |
| cl-gpu | [#63 — BLAS: dsdot loses finite products outside f32 range](https://github.com/communal-software/cl-gpu/issues/63) | Open | Known mixed-dot range correctness bug; no demonstrated main inference dependency. |
| cl-gpu | [#64 — BLAS: accept provably disjoint interleaved layouts and unique outputs](https://github.com/communal-software/cl-gpu/issues/64) | Open | GPU rejects valid interleaved layouts; needed for reliable CPU/GPU tensor parity. |
| cl-gpu | [#65 — BLAS tests: backend-neutral conformance and encoded bit-level coverage](https://github.com/communal-software/cl-gpu/issues/65) | Open | Portable backend conformance and exact half-bit checks support safe GPU expansion. |
| trivial-simd | [#6 — Add x86-64 AVX2 SIMD kernels](https://github.com/communal-software/trivial-simd/issues/6) | Open | Broad x86 throughput upgrade; benchmark real inference before committing kernel scope. |
| trivial-simd | [#31 — Harden CI against transient Quicklisp archive download failures](https://github.com/communal-software/trivial-simd/issues/31) | Open | Transient dependency setup failures prevent tests from running; keep retries bounded. |
| trivial-simd | [#33 — Evaluate full IEEE consistency for kernel arithmetic](https://github.com/communal-software/trivial-simd/issues/33) | Open | Numerical contract investigation; finite-only behavior is documented, not a universal bug claim. |
| trivial-simd | [#35 — Reduce boxing in CCL and ECL ARM64 scalar FMA kernels](https://github.com/communal-software/trivial-simd/issues/35) | Open | Secondary ECL/CCL FMA paths have a measured large boxing penalty. |
| trivial-simd | [#55 — CCL x86-64 1.13 drops (setf (aref array CONSTANT) complex) on (complex single-float) arrays](https://github.com/communal-software/trivial-simd/issues/55) | Open | Secondary CCL compiler correctness audit; workarounds exist, wider exposure needs checking. |
| trivial-simd | [#56 — Reduce remaining x86-64 CCL test setup and lifetime-stress overhead](https://github.com/communal-software/trivial-simd/issues/56) | Open | Secondary CCL/Rosetta validation reliability; distinguish host crashes from library failures. |
| trivial-simd | [#58 — Tune multi-pass kernel planning, native setup and row execution](https://github.com/communal-software/trivial-simd/issues/58) | Open | Inference-stage multi-pass setup/recomputation can be material; measure whole stages. |
| trivial-simd | [#59 — CCL ARM64 typed complex-double arithmetic drops an operand](https://github.com/communal-software/trivial-simd/issues/59) | Open | Secondary CCL compiler correctness audit; workarounds exist, wider exposure needs checking. |
| trivial-simd | [#60 — Investigate intermittent Rosetta host assertion in CCL x86-64 (library-independent reproducer)](https://github.com/communal-software/trivial-simd/issues/60) | Open | Secondary CCL/Rosetta validation reliability; distinguish host crashes from library failures. |
| trivial-simd | [#74 — Tune encoded matmul for larger activation batches](https://github.com/communal-software/trivial-simd/issues/74) | Open | Measured large-batch encoded matmul gap; retain tiled BLAS dispatch and small-batch wins. |

### P3 — deferred work

| Repository | Ticket | State | Planning note |
|---|---|---|---|
| cl-inference | [#20 — GGUF reader: Windows file mapping](https://github.com/communal-software/cl-inference/issues/20) | Open | Windows/BSD are explicitly low priority, including their CI requirements. |
| cl-inference | [#21 — GGUF reader: verify BSD file mapping](https://github.com/communal-software/cl-inference/issues/21) | Open | Windows/BSD are explicitly low priority, including their CI requirements. |
| cl-inference | [#22 — GGUF reader: faster string decoding](https://github.com/communal-software/cl-inference/issues/22) | Open | Load-time string-copy optimization; no added model capability. |
| cl-inference | [#25 — client: json-get is O(n^2) when building large JSON objects](https://github.com/communal-software/cl-inference/issues/25) | Open | Existing implementation is adequate at human request rates; scale only with evidence. |
| cl-inference | [#26 — client: one watcher thread per completion deadline](https://github.com/communal-software/cl-inference/issues/26) | Open | Existing implementation is adequate at human request rates; scale only with evidence. |
| cl-inference | [#27 — client: drop the usocket connect-refusal workaround](https://github.com/communal-software/cl-inference/issues/27) | Open | Working refusal workaround exists; removal depends on an upstream usocket fix. |
| cl-inference | [#39 — KV cache: bulk copy of new keys and values](https://github.com/communal-software/cl-inference/issues/39) | Open | Targeted decode-loop optimization; promote only with end-to-end profiling. |
| cl-inference | [#40 — Sampler: partial selection instead of a full-vocab sort](https://github.com/communal-software/cl-inference/issues/40) | Open | Targeted decode-loop optimization; promote only with end-to-end profiling. |
| cl-tensor | [#3 — Autograd](https://github.com/communal-software/cl-tensor/issues/3) | Open | Training/autograd is outside the immediate inference/model-coverage goal. |
| cl-tensor | [#5 — Reduce workspace for general output-layout validation](https://github.com/communal-software/cl-tensor/issues/5) | Open | Valid irregular layouts work; reduce exact-validation workspace with upstream coordination. |
| cl-tensor | [#7 — Reduce workspace for irregular axis reductions](https://github.com/communal-software/cl-tensor/issues/7) | Open | Bound scratch/allocation while preserving aliases and failure behavior; promote for measured model OOMs. |
| cl-tensor | [#8 — Reduce normalization workspace and portable f64 stage allocation](https://github.com/communal-software/cl-tensor/issues/8) | Open | Bound scratch/allocation while preserving aliases and failure behavior; promote for measured model OOMs. |
| cl-tensor | [#10 — Bound remaining accessible-storage fallback packing](https://github.com/communal-software/cl-tensor/issues/10) | Open | Bound scratch/allocation while preserving aliases and failure behavior; promote for measured model OOMs. |
| cl-tensor | [#11 — Bound take! scratch for disjoint destinations](https://github.com/communal-software/cl-tensor/issues/11) | Open | Bound scratch/allocation while preserving aliases and failure behavior; promote for measured model OOMs. |
| cl-tensor | [#13 — Bound fusion result staging and irregular reduction-row packing](https://github.com/communal-software/cl-tensor/issues/13) | Open | Bound scratch/allocation while preserving aliases and failure behavior; promote for measured model OOMs. |
| cl-tensor | [#16 — Support integer and complex tensor expression fusion](https://github.com/communal-software/cl-tensor/issues/16) | Open | Integer/complex fusion is not a demonstrated transformer requirement. |
| cl-tensor | [#20 — Allow per-call executor overrides](https://github.com/communal-software/cl-tensor/issues/20) | Open | Dynamic executor binding already works; keyword selection is convenience. |
| cl-gpu | [#27 — Source locations for define-kernel compiled from an editor buffer (C-c C-c)](https://github.com/communal-software/cl-gpu/issues/27) | Open | Editor diagnostic convenience; no execution or model-coverage change. |
| cl-gpu | [#33 — Support ABCL on the JVM](https://github.com/communal-software/cl-gpu/issues/33) | Open | Lisp implementations outside SBCL/ECL/CCL are explicitly low priority. |
| cl-gpu | [#34 — Support Clasp](https://github.com/communal-software/cl-gpu/issues/34) | Open | Lisp implementations outside SBCL/ECL/CCL are explicitly low priority. |
| cl-gpu | [#35 — Support LispWorks](https://github.com/communal-software/cl-gpu/issues/35) | Open | Lisp implementations outside SBCL/ECL/CCL are explicitly low priority. |
| cl-gpu | [#36 — Support Allegro CL](https://github.com/communal-software/cl-gpu/issues/36) | Open | Lisp implementations outside SBCL/ECL/CCL are explicitly low priority. |
| cl-gpu | [#39 — GPU BLAS: complex routines](https://github.com/communal-software/cl-gpu/issues/39) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#41 — GPU BLAS: rotations (srot, srotm, srotg, srotmg)](https://github.com/communal-software/cl-gpu/issues/41) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#43 — GPU BLAS: reuse the reduction partials buffer](https://github.com/communal-software/cl-gpu/issues/43) | Open | Targeted allocation, transfer, launch or alignment tuning; require an inference profile before promotion. |
| cl-gpu | [#46 — GPU BLAS: cache MPS kernel objects](https://github.com/communal-software/cl-gpu/issues/46) | Open | Targeted allocation, transfer, launch or alignment tuning; require an inference profile before promotion. |
| cl-gpu | [#48 — Metal: tune or adapt the command-buffer commit interval](https://github.com/communal-software/cl-gpu/issues/48) | Open | Targeted allocation, transfer, launch or alignment tuning; require an inference profile before promotion. |
| cl-gpu | [#50 — Metal: one stream lock per device serialises encoding](https://github.com/communal-software/cl-gpu/issues/50) | Open | Targeted allocation, transfer, launch or alignment tuning; require an inference profile before promotion. |
| cl-gpu | [#55 — GPU BLAS: f64 Level 1 variants (daxpy ... idamax)](https://github.com/communal-software/cl-gpu/issues/55) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#56 — Metal: emulate f64 in software](https://github.com/communal-software/cl-gpu/issues/56) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#60 — Metal: typed source loops for array transfers](https://github.com/communal-software/cl-gpu/issues/60) | Open | Targeted allocation, transfer, launch or alignment tuning; require an inference profile before promotion. |
| cl-gpu | [#61 — Metal: vectorised f32 conversion for f16 and bf16 transfers](https://github.com/communal-software/cl-gpu/issues/61) | Open | Targeted allocation, transfer, launch or alignment tuning; require an inference profile before promotion. |
| cl-gpu | [#62 — Faster nested vector-element transfers (to-device / from-device)](https://github.com/communal-software/cl-gpu/issues/62) | Open | Targeted allocation, transfer, launch or alignment tuning; require an inference profile before promotion. |
| cl-gpu | [#66 — BLAS: add a convenience system matching the CPU slice and matrix wrappers](https://github.com/communal-software/cl-gpu/issues/66) | Open | Wrapper parity is convenience; strict calls already exist. |
| cl-gpu | [#67 — BLAS Level 2: add dense f32 rank-one update sger](https://github.com/communal-software/cl-gpu/issues/67) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#68 — BLAS: add dense f32 symmetric matrix-vector, multiply and rank updates](https://github.com/communal-software/cl-gpu/issues/68) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#69 — BLAS: add dense f32 triangular multiply and solve routines](https://github.com/communal-software/cl-gpu/issues/69) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#70 — BLAS: add band and packed matrix storage with f32 Level 2 routines](https://github.com/communal-software/cl-gpu/issues/70) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#71 — BLAS: f64 GEMV, GEMM and batches on capable devices](https://github.com/communal-software/cl-gpu/issues/71) | Open | General BLAS/f64 breadth has no demonstrated requirement for the selected inference models. |
| cl-gpu | [#72 — Metal: query the real constant-space buffer offset alignment](https://github.com/communal-software/cl-gpu/issues/72) | Open | Targeted allocation, transfer, launch or alignment tuning; require an inference profile before promotion. |
| trivial-simd | [#7 — Add x86-64 AVX-512 SIMD kernels](https://github.com/communal-software/trivial-simd/issues/7) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#8 — Add ARM64 SVE/SVE2 SIMD kernels](https://github.com/communal-software/trivial-simd/issues/8) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#9 — Support 32-bit x86 SSE2](https://github.com/communal-software/trivial-simd/issues/9) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#10 — Support ARMv7 NEON](https://github.com/communal-software/trivial-simd/issues/10) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#11 — Support RISC-V Vector SIMD](https://github.com/communal-software/trivial-simd/issues/11) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#12 — Support PowerPC VSX SIMD](https://github.com/communal-software/trivial-simd/issues/12) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#13 — Support WebAssembly SIMD128](https://github.com/communal-software/trivial-simd/issues/13) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#14 — Support FreeBSD](https://github.com/communal-software/trivial-simd/issues/14) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#15 — Support OpenBSD](https://github.com/communal-software/trivial-simd/issues/15) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#16 — Support NetBSD](https://github.com/communal-software/trivial-simd/issues/16) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#17 — Support Android](https://github.com/communal-software/trivial-simd/issues/17) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#18 — Support iOS](https://github.com/communal-software/trivial-simd/issues/18) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#19 — Support ABCL on the JVM](https://github.com/communal-software/trivial-simd/issues/19) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#20 — Support CLISP](https://github.com/communal-software/trivial-simd/issues/20) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#21 — Support Clasp](https://github.com/communal-software/trivial-simd/issues/21) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#22 — Support CMUCL](https://github.com/communal-software/trivial-simd/issues/22) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#23 — Support MKCL](https://github.com/communal-software/trivial-simd/issues/23) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#24 — Support LispWorks](https://github.com/communal-software/trivial-simd/issues/24) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#25 — Support Allegro CL](https://github.com/communal-software/trivial-simd/issues/25) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#26 — Support JSCL in Node and browsers](https://github.com/communal-software/trivial-simd/issues/26) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#27 — Support CCL on Windows](https://github.com/communal-software/trivial-simd/issues/27) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#28 — Support ECL on Windows](https://github.com/communal-software/trivial-simd/issues/28) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#29 — Support GCL](https://github.com/communal-software/trivial-simd/issues/29) | Open | Optional ISA/OS/Lisp expansion; Windows stays low even for ECL/CCL. |
| trivial-simd | [#30 — Investigate SBCL ARM64 in-process SIMD backend](https://github.com/communal-software/trivial-simd/issues/30) | Open | Supported C NEON path exists; in-process SIMD is an alternative backend. |
| trivial-simd | [#32 — Extend kernel scratch addressing beyond 16-bit indexes](https://github.com/communal-software/trivial-simd/issues/32) | Open | 65,536 scratch-block ceiling has no demonstrated inference workload. |
| trivial-simd | [#34 — Profile and reduce native kernel call setup for short vectors](https://github.com/communal-software/trivial-simd/issues/34) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#36 — Evaluate concurrent ECL runner compilation for different signatures](https://github.com/communal-software/trivial-simd/issues/36) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#37 — Reduce complex SIMD transfer overhead](https://github.com/communal-software/trivial-simd/issues/37) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#38 — Native complex (c/z) BLAS Level 2 and 3 kernels](https://github.com/communal-software/trivial-simd/issues/38) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#39 — Native band, packed and symmetric BLAS Level 2 kernels](https://github.com/communal-software/trivial-simd/issues/39) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#40 — Native BLAS: fused trsm micro-kernel with packed inverted-diagonal A](https://github.com/communal-software/trivial-simd/issues/40) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#41 — Native BLAS: keep the fused trsm tile solve in registers](https://github.com/communal-software/trivial-simd/issues/41) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#42 — Native BLAS: fused trmm micro-kernel](https://github.com/communal-software/trivial-simd/issues/42) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#43 — Native BLAS: column-major trsm/trmm copy into the dense work buffer is slow](https://github.com/communal-software/trivial-simd/issues/43) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#44 — x86 trsm: fused kernel is slower relative to gemm for f32 AVX+FMA and at 64x64](https://github.com/communal-software/trivial-simd/issues/44) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#45 — Native complex BLAS level-1 reductions (scasum, dzasum, icamax, izamax, scnrm2, dznrm2)](https://github.com/communal-software/trivial-simd/issues/45) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#46 — SIMD argmin/argmax/asum: packed native lanes, sb-simd, native integers](https://github.com/communal-software/trivial-simd/issues/46) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#47 — SBCL f32 nrm2 kernel uses a scalar loop](https://github.com/communal-software/trivial-simd/issues/47) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#48 — Fuse native kernel reducers into the register VM](https://github.com/communal-software/trivial-simd/issues/48) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#50 — Extend and tune the inline small-array path](https://github.com/communal-software/trivial-simd/issues/50) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#51 — Compiler style-warnings on SBCL ARM64 and CCL](https://github.com/communal-software/trivial-simd/issues/51) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#52 — x86-64 f32 to f16 takes a scalar path for any block of eight with a subnormal result](https://github.com/communal-software/trivial-simd/issues/52) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#53 — Use F16C/AVX2 for f16 and bf16 on x86-64, and hardware f16 on Windows ARM64](https://github.com/communal-software/trivial-simd/issues/53) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#54 — Lisp f16/bf16 fallback speed on non-SBCL Lisps](https://github.com/communal-software/trivial-simd/issues/54) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#57 — Packed SBCL execution for declared kernel inputs](https://github.com/communal-software/trivial-simd/issues/57) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#62 — Native integer and complex row reductions](https://github.com/communal-software/trivial-simd/issues/62) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#63 — Prove N-D output uniqueness without element-count workspace](https://github.com/communal-software/trivial-simd/issues/63) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#65 — Reduce N-D metadata setup for short spans](https://github.com/communal-software/trivial-simd/issues/65) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#66 — Reduce exact GEMM validation search for difficult layouts](https://github.com/communal-software/trivial-simd/issues/66) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#67 — Extend packed conversion coverage beyond common integer and encoded paths](https://github.com/communal-software/trivial-simd/issues/67) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#68 — Extend SBCL packed masks and conversion primitives](https://github.com/communal-software/trivial-simd/issues/68) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |
| trivial-simd | [#77 — Specialize common native stride layouts for packed memory access](https://github.com/communal-software/trivial-simd/issues/77) | Open | Specialized performance/cleanup work with no demonstrated model-coverage blocker; require complete-call evidence. |

[^scope]: Inventory comes from `gh issue list --state open --limit 1000` for `communal-software/cl-inference`, `cl-tensor`, `cl-gpu` and `trivial-simd`, including bodies and discussion. SourceHut migration text is historical context; GitHub numbers and current states are used throughout. Local READMEs, support/limitations pages, tensor executor docs and the closed threading prerequisite supply targeted cross-checks. Rankings do not claim model-level speedups from primitive benchmarks.

[^selection81]: Native scalar/SIMD harnesses pass on ARM64 and x86-64/Rosetta. Complete trivial-simd and cl-tensor suites pass on ARM64 SBCL, CCL and ECL; the x86-64 SBCL suite also passes. The isolated trunk candidate passes 17 native tests, 876,870 upstream SBCL checks and 136,268 cl-tensor checks; its tracked files match the integrated commit. Implementations: [trivial-simd](https://github.com/communal-software/trivial-simd/commit/cd61d43c2986c05ac32503d7619b2e7b67d1b584) and [cl-tensor](https://github.com/communal-software/cl-tensor/commit/81272fa7ae765673ddaf43e0f984a1b035274ff4). The [selection profile](https://github.com/communal-software/trivial-simd/blob/cd61d43c2986c05ac32503d7619b2e7b67d1b584/docs/kernel-performance.md#lazy-selection) records the masking cost.

[^selected-review]: Planning baselines were cl-inference `aca63cf` and cl-tensor `81272fa`. Both GitHub issue bodies/discussions and prerequisite states were checked. Validation includes the native renderers/parsers, pinned reference generator, executor contract, fixture suites and real-model acceptance.

[^default-tokenizer]: Reference revision `8bb3cf0d216a5ae48360179f2931daf284be4d01`. The [stdlib-only generator](../tools/default-tokenizer-reference.py) compiles the actual C++ splitter and verifies token IDs for absent, empty and explicit `default` pre-types using the pinned tokenizer executable. Fifty selected inputs and 100 seeded mixed-character inputs are retained as offline fixtures. Regeneration with `--check` matches the tracked expectations.
