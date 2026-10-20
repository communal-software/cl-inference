# Continuous integration

GitHub Actions runs offline client, CPU and example checks on Linux/SBCL. Manual runs and tags also require Qwen3 F16 model acceptance.

## Triggers

| Event | Offline checks | Qwen3 acceptance |
|---|---|---|
| Branch push or pull request with code changes | Yes | No |
| Changes only under `docs/`, Markdown files or `LICENSE` | No | No |
| Tag push | Yes | Yes |
| Manual `CI` workflow run | Yes | Yes |

New runs cancel older runs of the same workflow for the same PR or ref. Manual and tag runs execute regardless of changed paths. The separate `ql-dist` workflow notifies the Quicklisp distribution on tags and manual dispatch.

## Offline checks

The client runs in a fresh Lisp process and verifies that neither engine nor tensor packages load. CPU fixtures, acceptance-runner regression tests and runnable examples run afterward. Python tests cover reference preparation without downloading models or building an oracle.

The jobs use Ubuntu 24.04, SBCL 2.6.8 and native trivial-simd kernels. Both jobs use the dependency pins in `.github/actions/setup-cpu/action.yml`:[^pins]

| Dependency | Commit |
|---|---|
| cl-tensor | `a7f765911268adfce9668ee8b4356f1ab238b01f` |
| trivial-simd | `19d734f8806e07d5ce5b806e90b44621f4347146` |

Local development setup is described in [Getting started](getting-started.md#install).

## Model acceptance

After offline checks pass, the acceptance job downloads the [pinned Qwen3 F16 model](acceptance.md#qwen3-inputs), checks its SHA256 and builds the tracked oracle dumper against pinned CPU-only llama.cpp. Preparation creates three references: plain completion, thinking chat and non-thinking chat.

Run the same checks locally with the CPU dependencies installed and Quicklisp loaded:

```sh
python3 tests/acceptance/prepare-qwen.py
CL_INFERENCE_CI_SUITE=acceptance sbcl --dynamic-space-size 4096 \
  --non-interactive --load tests/ci.lisp
```

The test runner selects all three cases plus model metadata. It requires exact prompt and greedy token IDs, layer traces and final logits within the existing [F16 bounds](acceptance.md#llama-reference-checks). Missing inputs, mismatched model hashes, skipped checks and test failures fail the job.

Models, builds and generated references stay under gitignored `reference/`. Each reference has provenance containing the model URL/hash, llama.cpp revision, dumper hash, prompt settings and output hashes. Preparation does not rewrite tracked manifests.

## Diagnostics

The fixture job has a 30-minute timeout; acceptance has 90 minutes. Failed jobs upload logs. Acceptance failures also upload the generated oracle outputs and provenance, excluding model weights. The workflow needs no secrets and uses read-only repository permissions.

The workflow can be checked locally with `actionlint .github/workflows/ci.yml`. The [testing guide](testing.md) lists individual Lisp suites and offline Python commands.

## Limitations

Dependency caching is disabled, so fixture and acceptance jobs repeat the full toolchain setup. [Dependency caching](https://github.com/communal-software/cl-inference/issues/78) tracks this runtime improvement.

[^pins]: Update the shared action and this table together when changing dependency revisions. Other Lisp libraries come from the setup action's Quicklisp environment.
