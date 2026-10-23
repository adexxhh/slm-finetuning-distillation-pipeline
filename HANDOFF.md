# Project continuation handoff

## Objective
Finish and substantiate the three resume projects: MCP database auditor, targeted code editor, and SQL QLoRA pipeline. Preserve their central ideas, repair real implementation gaps, collect reproducible evidence, and prepare the owner to explain the engineering.
Do not describe the projects as perfect, production-ready, or fully benchmarked.

## Working copies and source
Local repair clones are E:/project-repairs/mcp-database-auditor, autonomous-code-editor, and slm-finetuning-distillation-pipeline. The local Python environment is E:/project-repairs/.venv.
GitHub repositories are under github.com/adexxhh with the same names. Each repository has README.md, OWNERSHIP.md, and real-run terminal PNG/GIF/SVG demos with transcripts.
Use HANDOFF.md plus the current git log as the entry point for continuation. Local Colab source archives are snapshots: pushing GitHub changes does not update a running notebook.

## Current SQL experiment
The Qwen3-8B NF4 QLoRA run completed 100 optimizer steps (0.8 epoch) on a Tesla T4. Training took about 85.3 minutes; final recorded validation loss is 0.26359. Full step-90/100 checkpoint evidence and approximately 43.65 million adapter parameters are present in the original archive at E:/Downloads/sql-8b-results.zip.

The frozen 100-question evaluation has 70 base and 71 adapter execution matches. Paired outcomes are 61 both, 20 neither, 10 adapter-only and 9 base-only. This does not establish a reliable quality improvement. A database-cluster bootstrap gives an exploratory difference interval of approximately -10.74 to +12.09 points.

A completed experiment report, frozen predictions and provenance live in the SQL repo at docs/EXPERIMENT.md and experiments/results/t4-qlora-001. No model weights are committed. CPU replay verifies evidence/database hashes and reproduces original scores with only SQLGlot installed. A fresh isolated environment replayed the archive and an independent fixture directory; repeated diagnostics matched byte for byte.

Column-order diagnostics flag three base and two adapter cases without changing the original scores. Seeded in-memory numeric variants expose one accidental original match per model. These are sensitivity diagnostics, not official Spider test-suite accuracy or proof of general semantic correctness. Dataset-derived excerpts include attribution.

The current bounded experiment is complete. Next optional research iterations should use validation and a fresh final holdout; do not tune repeatedly on this inspected cohort. Repeating GPU training across seeds or environments has not been performed.

## Persistent state and recovery
The revised SQL training cell asks the user to mount Google Drive and uses:
MyDrive/ai-project-experiments/sql-8b-run-001/adapter
Complete checkpoints contain complete.json, trainer_state.json, optimizer.pt, scheduler.pt, adapter files and RNG state. Auto-resume chooses a fully marked checkpoint and validates training/data fingerprints. Partial or incompatible checkpoints must not be silently resumed.
training_identity.json guards the dataset/configuration; changing training settings requires a new output directory. A local tiny CPU interruption test stopped at step 2 of 4, restored the optimizer/scheduler, advanced to step 4 and changed adapter weights. See docs/evidence/resume-verification.json in the SQL repo.
Google Drive errors such as credential propagation failure occur before training; use the notebook's same Google account and complete authorization manually. Do not request passwords or tokens.
Checkpoint persistence and the full CUDA/Drive recovery flow have not yet been independently verified end to end.
Baseline predictions/splits can remain runtime-local until the training cell copies them to Drive. A runtime reset may require model download and preparation again. The entire pipeline is not yet automatically resumable stage by stage.
Download sql-8b-results.zip before session expiry. The notebook also copies the completed archive to the run directory in Drive.

## Latest local verification
- Auditor: 58 tests passed; one opt-in PostgreSQL test skipped without a disposable server. Actual MCP stdio client and SQLite timing benchmark passed.
- Editor: 12 tests passed; new Redis integration test skipped on Windows/no disposable service. Notebook code cells compile.
- SQL: 28 tests passed, including result comparison, numeric variants and frozen-evidence checks. Tiny CPU training/export/recovery and fresh-environment experiment replay also passed.
- Linux PostgreSQL and Redis service jobs passed on GitHub Actions; see the verified runs below.
The Windows TCP fake was inadequate for ARQ INFO commands and POSIX lifecycle behavior; that failed attempt is not real Redis evidence.

## Claims and evaluation boundaries
Withdrawn unsupported original metrics: editor 45% general token reduction and 92% repair rate; SQL 80% model execution accuracy; hardcoded model latency/cost comparisons; mocked dispatch figures presented as whole-system performance.
The old SQL evaluator reused gold queries as model predictions. The repaired evaluator accepts independent predictions and runs them on populated read-only databases.
Current SQL experiment is a filtered, database-disjoint public Spider development subset. It reports single-database execution matching, not official Spider test-suite accuracy.
Local auditor SQLite schema timing: median about 14.80 ms / p95 15.80 ms, 30 runs. This is not a PostgreSQL latency guarantee.
Editor token comparison is one toy output-only payload example (23 versus 9 tokens), not a broad performance result.
Git worktrees isolate source changes, not the operating system. Keep generated-code probes in a disposable runtime without personal Drive mounted.

## Verified GitHub Actions runs
- Auditor PostgreSQL and SQLite/MCP jobs: https://github.com/adexxhh/mcp-database-auditor/actions/runs/39961281678 (source be6b4e9).
- Editor CPU and real Redis/ARQ jobs: https://github.com/adexxhh/autonomous-code-editor/actions/runs/39961303101 (source 672c77f).
- SQL CPU contracts: https://github.com/adexxhh/slm-finetuning-distillation-pipeline/actions/runs/39961314334 (source 00286c7).
All listed jobs completed successfully. Each repo preserves its API job/step results in docs/evidence/github-actions.json. These results apply to those immutable source commits; later documentation commits do not establish a new runtime experiment.
Real PostgreSQL pooled read-only and real Redis queued completion/cancellation/scoped-event checks are now verified on Linux. Colab service notebooks remain optional reproduction tools. Docker Compose as a complete deployment, actual editor LLM repair results, and broader independent SQL model-quality experiments remain future work.

## Next actions
1. Read the completed SQL experiment report and frozen evidence. Preserve the original results ZIP and Drive checkpoints.
2. Use the published CPU replay to verify scores and diagnostics. Keep any later training experiment separate from the inspected final cohort.
3. Preserve the passing real PostgreSQL/Redis CI checks and inspect future failures; the CPU Colab notebook is now optional reproduction.
4. Run the editor's actual-model probes in a separate Drive-free GPU session; retain every proposal, test failure and diff.
5. Continue integrating the editor's code-repair loop with its asynchronous service; the original service still has a separate search workflow.
6. Broaden representative repair evaluation, implement OS-level execution isolation for untrusted service use, and verify actual SSE/worker concurrency before claiming production readiness.
7. Explain changes using each OWNERSHIP.md. User understanding and fresh reproductions are part of interview readiness.

## Rebuild cloud source bundles
Clone the three repositories as sibling directories. From the SQL repo run:
python scripts/build_source_bundles.py --output-dir ../
This creates sql-8b-source.zip and remaining-projects-source.zip from clean committed source and records commit SHAs. No weights, checkpoints, credentials or notebook outputs are included.
The SQL notebook lives in experiments/colab_8b_qlora.ipynb.
The CPU service notebook lives in mcp-database-auditor/scripts/colab_service_checks.ipynb.
The editor GPU notebook lives in autonomous-code-editor/benchmarks/colab_model_repairs.ipynb.

## This repository
Implemented strict SQL validation, seeded separate splits, completion-only LoRA loss, real HF/PEFT inference, independent SQL execution evaluation, guarded Trainer resume and real adapter export. The completed notebook used Qwen3-8B, 100 optimizer steps, batch 1, accumulation 8 and max sequence 2048. Buffered SSE is not live token streaming; external GGUF conversion and representative quality claims remain unverified.
