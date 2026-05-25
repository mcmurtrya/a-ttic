# Pipeline Run Brief — for autonomous agent

> **Path note.** This brief was written for the cloud-rental Ubuntu host
> that produced the original Phase A results. Wherever `/home/ubuntu/`
> appears below, substitute your repo root (e.g. `$HOME/ttic_embeddings`
> or `/home/<user>/projects/.../ttic_embeddings`). The paths are not
> magic; they describe the layout of the machine that ran the pipeline,
> not a requirement of the pipeline itself.

> **How to use this file**: After restarting Claude with `--dangerously-skip-permissions`, paste this single instruction:
>
> *"Read `<repo-root>/RUN_PIPELINE.md` and spawn a `general-purpose` agent in the background using the AGENT BRIEF section below as the agent's prompt. Then end your turn — you'll be notified when the agent finishes."*

---

## AGENT BRIEF

You are running the full TTIC Embeddings training+analysis pipeline end-to-end on a fresh GPU instance, then committing results and shutting down. The user (amcmurtry@uchicago.edu) is offline; you operate autonomously.

### Permissions context
This session was launched with `--dangerously-skip-permissions`, so policy denials should not occur. Two prior agent attempts were blocked by an over-tight `.claude/settings.local.json`; the user has since widened it and restarted the session.

### Environment (verified, do not re-check)
- Repo: `/home/ubuntu/ttic_embeddings` (branch `main`, clean as of 2026-05-10)
- 2× NVIDIA A100-SXM4-80GB GPUs, idle, on host `instance-92zknf1f-main`
- 147 GB free on `/`, no venv yet, no data yet
- `uv` at `/usr/local/bin/uv`, `sudo` is passwordless
- Git remote: `git@github.com:mcmurtrya/a-ttic.git` (SSH push works)
- No git `user.name`/`user.email` set yet → set them locally on the repo before committing
- `.gitignore` excludes `/captions/`, `/checkpoints/`, `/data/`, `/wandb/`, `/features/` — you must `git add -f` whichever results you want to commit
- `pyproject.toml` pins `en-core-web-lg` directly in the `dev` extra, so `make install-dev` installs the spaCy model automatically — no separate `python -m spacy download` needed

### What "the pipeline" is
`bash scripts/run_full_pipeline.sh` — read it first. Phases: smoke test → train 12 adaptors (4 encoders × 3 seeds) on 2 GPUs in parallel pairs → generate captions (seed 1) → score metrics → statistical analysis (beam + nucleus) → linear probes → CIDEr/SPICE quality precondition. Phase logs go to `logs/<timestamp>/*.log`. Resume-safe (training and generation auto-resume on re-run).

Estimated wall time: ~10–15 hours.

### Plan

#### Phase A — Setup
1. `cd /home/ubuntu/ttic_embeddings`
2. `make install-dev` (uv sync with dev + caption-quality extras + spaCy model + NLTK assets)
3. Export data roots and download:
   ```
   export COCO_ROOT=$HOME/data/coco
   export VG_ROOT=$HOME/data/vg
   mkdir -p "$COCO_ROOT" "$VG_ROOT"
   make data    # ~26 GB
   ```
   Persist these env vars in `/home/ubuntu/ttic_embeddings/.env.pipeline` and `source` it for every later step. Do NOT modify `~/.bashrc`.

#### Phase B — Pipeline
4. Run smoke test: `uv run python scripts/00_smoke_test.py`. If it fails, fix before proceeding.
5. Launch the full pipeline detached:
   ```
   nohup bash -c 'export COCO_ROOT=$HOME/data/coco; export VG_ROOT=$HOME/data/vg; bash scripts/run_full_pipeline.sh' > pipeline.out 2>&1 &
   ```
   Save the PID. Use `Bash` with `run_in_background=true` to launch, then poll every 10–30 min via `tail` of the latest `logs/<timestamp>/*.log` and `pipeline.out`. Don't poll faster than necessary.

#### Phase C — Monitoring and intervention
- **OOM / CUDA errors**: investigate, reduce batch size in `configs/<encoder>.yaml`, re-run pipeline (it resumes).
- **Stuck jobs** (no log progress >30 min, GPU util 0% via `nvidia-smi`): kill the python PID, inspect log, fix, restart.
- **Disk space**: `df -h /home/ubuntu` periodically.
- **NaN loss / divergence**: visible in train logs; document but don't block other encoders.
- **Pipeline non-zero exit**: re-run; resume-safe. Up to 3 retries for transient failures. After 3 attempts, stop and write a failure report (Phase E).

Don't edit code unless a real bug surfaces. The repo just merged a code-review pass; trust it. If you DO edit code, make a single focused commit on a fix branch (NOT main) explaining the bug + fix before continuing.

#### Phase D — Commit results (only on success)
After pipeline prints "Pipeline complete." and exits 0:

1. Set git identity locally:
   ```
   git -C /home/ubuntu/ttic_embeddings config user.name "Adam McMurtry"
   git -C /home/ubuntu/ttic_embeddings config user.email "amcmurtry@uchicago.edu"
   ```
2. Create branch `pipeline-results-YYYYMMDD` from `main`.
3. Force-add (gitignored paths):
   - `logs/<timestamp>/` (text logs)
   - `captions/*.jsonl`
   - `captions/*.csv` (scores, analysis, probe_results, quality_results)
   - `captions/quality_summary.txt`
   - `checkpoints/*/adaptor_best.pt` **only if** total `du -sh checkpoints/` < 500 MB AND no single `.pt` > 90 MB. Otherwise skip and note in commit message.
4. Single commit with descriptive message; push: `git push -u origin pipeline-results-YYYYMMDD`. Do NOT push to `main`. Report branch + SHA in final summary.

#### Phase E — Shutdown
- **Success**: write `/home/ubuntu/pipeline_summary.txt` (start/end times, log dir, branch + commit SHA, anomalies). Then `sudo shutdown -h now`.
- **Unrecoverable failure** (3 retries exhausted): write `/home/ubuntu/pipeline_summary.txt` with failure context + log paths + last commands tried. **Do NOT shut down.** Stop.

### Operating notes
- `Bash` with `run_in_background=true` for the long pipeline. Monitor via `tail` (cheap) or load `Monitor` tool via ToolSearch (`select:Monitor`).
- Append-only running notes at `/home/ubuntu/agent_notes.md` to recover state across compactions.
- Don't `git push --force`, push to `main`, `--no-verify`, amend, `git clean`, `git reset --hard`.
- Don't modify `~/.bashrc`, `~/.gitconfig`, `~/.ssh/`, `~/.claude/`.
- Pipeline already handles GPU pairing — don't reinvent.

### Final report should include
(a) success/failure, (b) branch + commit SHA if pushed, (c) summary path, (d) whether the instance was shut down.
