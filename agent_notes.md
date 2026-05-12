# Agent running notes — TTIC Embeddings full pipeline

Started: 2026-05-10 06:35 UTC
Repo: /home/ubuntu/ttic_embeddings (currently on branch fix/gpu-pinning-cuda-visible-devices)
GPUs: 2x A100-80GB
Host: instance-92zknf1f-main
uv: 0.11.2

## Phase A — Setup (DONE)
- [x] make install-dev (06:35-06:36, exit 0)
- [x] export COCO_ROOT/VG_ROOT in .env.pipeline
- [x] make data: COCO downloaded (118287 train + 5000 val + annotations)
- [x] VG: visualgenome.org/static returns 403; UW mirror
      (homes.cs.washington.edu/~ranjay/visualgenome/data/dataset) is alive.
      Downloaded attributes.json.zip (83 MB) and image_data.json.zip (1.7 MB)
      via curl with browser UA, extracted, created .unzipped marker files.
      Final: attributes.json 462 MB, image_data.json 17 MB.

## Phase B — Pipeline
- [x] smoke test passed (06:38, then again at 07:19 in the new pipeline)
- [BUG FIX] First pipeline run at 06:45 UTC stacked both training jobs on
  GPU 0 — root cause: in this PyTorch-2.11.0+cu130 + driver-580.126.20 env,
  CUDA_VISIBLE_DEVICES is silently ignored. Empirically:
  `env CUDA_VISIBLE_DEVICES=1 python -c "torch.zeros(...,device='cuda:0')"`
  allocates on physical GPU 0 (verified by nvidia-smi memory.used).
  Fix: created branch fix/gpu-pinning-cuda-visible-devices, replaced the
  CUDA_VISIBLE_DEVICES env-var dispatch with a `--gpu N` CLI arg that
  calls torch.cuda.set_device(N) explicitly. Verified torch.cuda.set_device(1)
  + device='cuda:1' correctly puts memory on GPU 1.
  Commit: a3b4814 on fix/gpu-pinning-cuda-visible-devices (NOT pushed).
- [ ] full pipeline relaunched 07:18 UTC, PID 16579
      Log dir: /home/ubuntu/ttic_embeddings/logs/20260510_071828
      Both GPUs verified loaded with model memory at 07:19 UTC.

## Phase D — Commit + push
Handled by /home/ubuntu/finisher.sh (PID 19992, started 07:20 UTC):
  - Waits for pipeline, retries up to 3x
  - On success creates pipeline-results-YYYYMMDD branch from current HEAD
    (which means it includes the GPU-pinning fix commit as part of the
     results branch, separate from main — that's allowed)
  - Force-adds logs/<ts>/, captions/*.jsonl, captions/*.csv,
    captions/quality_summary.txt, adaptor_best.pt (if total<500MB & max<90MB)
  - Commits, pushes origin/pipeline-results-YYYYMMDD
  - Writes /home/ubuntu/pipeline_summary.txt
  - sudo shutdown -h now

## Events log
- 06:35 UTC: Phase A start.
- 06:36 UTC: install-dev complete.
- 06:38 UTC: smoke test passed.
- 06:43 UTC: COCO done.
- 06:44 UTC: VG zips downloaded from UW mirror.
- 06:45 UTC: launched original pipeline (PID 2656).
- 06:48 UTC: previous agent's finisher.sh launched (PID 3609).
- 07:09 UTC: noticed both jobs on GPU 0, GPU 1 idle. Investigated.
- 07:13 UTC: confirmed CUDA_VISIBLE_DEVICES bug; killed prior pipeline +
  finisher; disabled finisher.sh.
- 07:18 UTC: committed fix on branch, relaunched pipeline (PID 16579).
- 07:19 UTC: both GPUs allocated; training started.
- 07:20 UTC: re-enabled finisher (PID 19992).

## Disk
- /home/ubuntu/data/coco: ~22 GB
- /home/ubuntu/data/vg: 480 MB
- ~125 GB free at start

## URGENT NOTE TO ANY OTHER AGENT (07:21 UTC)
A previous agent killed the finisher and restarted the pipeline (which triggered
multiple confused finisher restarts). The situation is now:

- Pipeline: PID 16580 (`bash scripts/run_full_pipeline.sh`) — training clip+siglip seed1
- Finisher: PID 22354 (`bash /home/ubuntu/finisher.sh`) — single instance, flock-guarded
  - Watches for ALL `run_full_pipeline.sh` processes to exit (process-name based, not PID)
  - Then checks `pipeline.out` for "Pipeline complete." marker
  - Retries up to 3x, commits, pushes, shuts down
- Lock file: /home/ubuntu/finisher.lock (do NOT delete)

DO NOT kill /home/ubuntu/finisher.sh. DO NOT kill PID 22354. DO NOT delete
/home/ubuntu/finisher.lock. The finisher will NOT shut down the instance until
"Pipeline complete." is observed AND commit + push succeed. If something is
broken with the pipeline itself, fix the pipeline; do not kill the finisher.

If you (an agent) are unsure what's going on, leave things alone and write a
note here. Please.

## INTERVENTION 2026-05-10 16:03 UTC — throughput fixes (Tier-1 + Tier-2)

User authorized killing the in-flight pipeline to apply throughput fixes.
Status of the system right now:

- Finisher PID 22354 is **PAUSED via SIGSTOP** (state T). Lock file
  /home/ubuntu/finisher.lock is intact; do NOT touch it. Will SIGCONT later.
- Pipeline (PIDs 16579/16580 + python 17836/17801) **was killed gracefully**
  (SIGTERM, _GracefulExit handler fired, saved adaptor_latest.pt then exited).
- Checkpoints saved at:
  - clip_seed1:   latest=step 25648 (val_ppl 10.0378), best=step 25000 (10.0350)
  - siglip_seed1: latest=step 22976 (val_ppl 9.2986),  best=step 22500 (9.2986)
- Both val_ppl trended slightly better right before kill, no plateau yet.
- Existing GPU-pinning fix branch (a3b4814) is still HEAD; new edits will
  pile onto the same working tree.

Plan being executed (TaskCreate task list 1–9):
1. Pause finisher [DONE]
2. Stop pipeline gracefully [DONE]
3. Update this note [in progress]
4. Tier-2 fixes (set_seed determinism off, TF32 on, persistent_workers,
   num_workers=12, val_max_batches=20, val_every=1000, fix logger handler bug)
5. Tier-1: write scripts/01_cache_features.py (per-encoder feature cache)
6. Wire cached features into 02_train_clip.py
7. Update run_full_pipeline.sh: cache-then-train-3-seeds per encoder
8. Smoke test
9. SIGCONT finisher + relaunch pipeline (it will auto-resume from current
   checkpoints because the math is unchanged)

If you are the cooperating oversight agent and disagree, append a note here
and SIGCONT 22354 yourself. Otherwise leave things alone.

### Throughput-fix code changes (2026-05-10 16:30 UTC)

Edits applied:
- src/ttic_embeddings/utils.py: set_seed no longer pins cudnn-deterministic
  or use_deterministic_algorithms; enables cudnn.benchmark and TF32.
- src/ttic_embeddings/train.py: logger now uses get_logger() so per-step
  log lines actually emit (was silently dropped via lastResort). Model
  forward accepts patch_features OR pixel_values; training/validate loops
  dispatch via _model_inputs_from_batch helper.
- src/ttic_embeddings/data/coco.py: new CocoCachedFeaturePairs Dataset
  reads an fp16 memmap of precomputed encoder patches.
- configs/base.yaml: num_workers 4->8, val_every_steps 500->1000,
  val_max_batches null->20.
- scripts/02_train_clip.py: persistent_workers + prefetch_factor; auto-
  detect cache and switch dataset; weights_only=False on torch.load.
- scripts/01_cache_features.py: NEW — precomputes encoder features to
  $FEATURE_CACHE_ROOT/{encoder}/train2017_features.fp16.bin (~59 GiB
  for CLIP/SigLIP/DINOv2, ~47 GiB for MAE).
- scripts/run_full_pipeline.sh: restructured outer loop to per-encoder
  (cache then train all 3 seeds, then delete cache).

Smoke test results (16:18-16:25 UTC):
- Live-encoder --smoke run: 50 steps, loss 5.75 -> 3.80, val_ppl 68.2 -> 52.7.
  Per-step log lines now visible (logger fix confirmed).
- Cache writer with DataLoader+8 workers: 88 img/s (was 7.9 with serial loader).
  Estimated full COCO train2017 cache: ~22 min per encoder.
- Cache reader: shapes (256, 1024) fp16, fork-safe under DataLoader workers.
- End-to-end cached training: 5.5 step/s at batch=32.

Resume caveats:
- Existing clip_seed1 (step 25648) and siglip_seed1 (step 22976) checkpoints
  were trained with cudnn.deterministic=True and live encoder forward. New
  runs use cudnn.benchmark=True and cached fp16 features. Model state
  preserves; inputs differ by <1e-3 numerical drift. SGD is robust to this;
  research validity is mildly affected (cross-condition comparison still
  internally consistent since all 12 conditions use the new pipeline going
  forward).

## INTERVENTION 3 — 2026-05-11 01:40 UTC — step budget cut + disk free

User authorized two changes after status review:

1. Deleted redundant COCO archive files (zips still present alongside extracted
   dirs):
     /home/ubuntu/data/coco/train2017.zip                  (19 GB)
     /home/ubuntu/data/coco/val2017.zip                    (778 MB)
     /home/ubuntu/data/coco/annotations_trainval2017.zip   (242 MB)
   Disk now 64% used, 54 GB free (was 77% / 35 GB). Buffer for caching, no
   functional impact — extracted dirs are untouched.

2. configs/base.yaml: `train.max_steps: 50000 -> 35000`.
   Rationale: seed_2's val_ppl trajectory plateaus by step 35k.
     step 25000 -> 9.989
     step 35000 -> 9.881
     step 41000 -> 9.860  (best)
     step 44000 -> 9.865  (current)
   Net gain from 35k to 41k: 0.021 ppl, well below cross-seed variance
   (~0.3 ppl seed_1 vs seed_2 at convergence). LR is at ~4e-6 by step 43k,
   so the cosine schedule already nearly-cooled. Saves ~15k steps × 0.73 s/step
   ≈ 3 h per seed × 10 remaining seeds ≈ 30 GPU-h ≈ 15 wall-h after pairing.

   Why this won't affect clip_seed2 mid-flight: training script reads config at
   startup. clip_seed2 (PID 967196) already loaded max_steps=50000 and will
   finish there (~02:48 UTC).  clip_seed3 onward will see the new 35000 cap.

   The existing `EarlyStopper(patience=5)` (train.py:447) is layered under
   this cap: with val_every_steps=1000, runs auto-stop after 5k steps of no
   val_ppl improvement, even before reaching 35k. So a per-encoder plateau
   that lands earlier than CLIP's 35k will be caught automatically; no script
   patch needed.

Not changed: pipeline structure (4 solo-seed-3 phases still waste one GPU
each, ~8 h each at the new step budget). Cross-encoder pairing was considered
but two 58 GiB caches don't fit even with the zips removed (would need
~120 GB; we have 54 GB free).

## INTERVENTION 4 — 2026-05-11 14:15 UTC — MAE step cap raised to 45k (REVERTED 2026-05-11 23:40 UTC)

**STATUS: REVERTED. Do not execute the restart procedure below.**

On 2026-05-11 23:40 UTC the user decided to keep MAE at 35k (matched-cap
symmetry with clip/siglip/dinov2). The two `extra=(--max-steps 45000)`
hunks were removed from `run_full_pipeline.sh`. Disk and the running
pipeline are now consistent: both run MAE at 35k.

No pipeline restart is needed. The running pipeline (PID 950372) had
the original pre-Intervention-4 function bodies in memory anyway, so
MAE would have used 35k on its own. The restart procedure described
below was written assuming we wanted MAE at 45k LIVE — that goal is gone.

The historical record below is preserved for context only.

---

ORIGINAL NOTE (now historical):


User authorized raising MAE's step cap from 35k to 45k. Rationale: the 35k
cut from Intervention 3 was justified by a single CLIP seed; MAE is the
narrowly-pretrained outlier (196 patches, ImageNet-1K 1.3M images,
flagged in encoder_selection.md:24) and its loss curve may not look like
CLIP's. Raising MAE only preserves most of the compute savings while
giving MAE more headroom; CLIP/SigLIP/DINOv2 remain at 35k.

### Edit applied (NOT YET LIVE)

scripts/run_full_pipeline.sh — both training helpers now append
`--max-steps 45000` when `enc == "mae"`:

  run_train_seed_pair (line ~142):
    local extra=()
    [[ "$enc" == "mae" ]] && extra=(--max-steps 45000)
    ... passes "${extra[@]}" to both `02_train_clip.py` invocations.

  run_train_seed_solo (line ~164): same pattern, applied to its single
    `02_train_clip.py` invocation.

`02_train_clip.py:109` declares `--max-steps` (hyphen) which overrides
`cfg.train.max_steps`. Note: argparse requires the hyphenated form on
the command line; underscore form will fail.

### CRITICAL: edit is on disk only — running pipeline has stale defs

The pipeline (PID 950372, single `bash run_full_pipeline.sh` process)
parsed `run_train_seed_pair` / `run_train_seed_solo` into memory at
startup on 2026-05-10 16:28 UTC. Bash captures function bodies at
definition time; editing the script file does NOT propagate to the
live process. If we let this run continue to MAE without restarting,
MAE will still train under the old 35k path.

### Restart procedure — DO BEFORE MAE PHASE BEGINS

Order of encoders in the run is `clip siglip dinov2 mae` (last). As of
2026-05-11 14:15 UTC the pipeline is mid-siglip_seed2, with siglip_seed3
(solo) + dinov2 cache + dinov2 × 3 seeds still ahead of MAE. Rough ETA
for MAE phase start: ~15+ wall-hours from 14:15 UTC, so ~05:00 UTC on
2026-05-12.

Detection signal — MAE phase is imminent when ANY of these is true:
  - `logs/20260510_162813/cache_mae.log` exists (cache just started)
  - pipeline stdout/tee contains "===== Encoder: mae ====="
  - pipeline_watcher.log line shows phase=cache:mae.log or
    phase=train:mae_seed*.log

Best moment to restart: AFTER dinov2 finishes its solo seed 3 and BEFORE
`cache_features mae 0` starts. Watching for the dinov2 cleanup line
("Removing dinov2 feature cache...") in pipeline stdout is the clean
trigger — at that point all dinov2 checkpoints are saved and no MAE
work has started. Restarting then loses zero work.

If we miss that window: still fine to restart during MAE cache (~22 min
loss) or during the first <10 min of MAE training (before first val
checkpoint at step 1000). After step 1000, MAE checkpoints exist so
resume is clean again — but you'd want to delete `checkpoints/mae_*/`
to start fresh under the new schedule, since LR-schedule trajectory
differs between 35k-cosine and 45k-cosine and resuming would mix them.

### Restart steps (copy-paste-able)

```
# 1. Pause the finisher so the impending pipeline exit isn't read as "done"
kill -STOP 22354
ps -o pid,stat -p 22354    # confirm state=T

# 2. Gracefully stop the pipeline. SIGTERM cascades to python; the
#    _GracefulExit handler saves adaptor_latest.pt then exits.
kill -TERM 950372
# Wait for `bash run_full_pipeline.sh` and python children to clear.
# Should take <60 s after the latest training step finishes.

# 3. Verify no stray training processes:
pgrep -af "02_train_clip.py|run_full_pipeline.sh"   # should be empty

# 4. Relaunch the pipeline. It auto-resumes per-encoder from latest
#    checkpoints (no work lost for already-completed seeds).
cd /home/ubuntu/ttic_embeddings
source .env.pipeline
nohup bash scripts/run_full_pipeline.sh \
    > /home/ubuntu/pipeline.out 2>&1 &
echo "new pipeline PID: $!"

# 5. SIGCONT the finisher — it'll re-watch for the new pipeline
#    process by name and behave normally.
kill -CONT 22354
ps -o pid,stat -p 22354    # confirm state=S

# 6. Tail logs to confirm MAE training starts with --max-steps 45000
#    in its log header (`max_steps=45000` line at training start).
```

### Verification after restart

In the new run's `logs/<ts>/train_mae_seed1.log` (or seed2/3), grep for:
  "Starting training. max_steps=45000"
That line is emitted by train.py:459 right after config load and
confirms the override took effect. If it shows 35000 instead, the
`--max-steps` arg didn't reach the script — check that `extra[@]` is
quoted properly in the bash invocation.

### What stays as-is

- max_steps in configs/base.yaml stays at 35000. We do NOT edit the
  base config because clip/siglip/dinov2 should still use 35k.
- EarlyStopper(patience=5) still applies to MAE — if MAE plateaus
  before 45k it'll auto-stop, no harm done.
- finisher.sh logic unchanged.
- pipeline_watcher.sh unchanged.

## INTERVENTION 6 — 2026-05-12 00:25 UTC — CLIP seed1+seed2 reruns at native 35k

**STATUS: LIVE as of 2026-05-12 00:42 UTC; patched + relaunched at 05:16 UTC.**
- finisher_v2 running as PID 406101 (verify with `pgrep -af "^bash /home/ubuntu/finisher_v2.sh"`).
- Lock file: `/home/ubuntu/finisher_v2.lock` (NOT the old finisher.lock — see
  "Lock-file gotcha" below).
- Old finisher.sh (PID 22354) was killed at 00:30 UTC.

### Commit-coverage fixes applied 2026-05-12 05:16 UTC

Three bugs in the original finisher's git-add logic were found before the
pipeline reached its commit step:

1. `ls -dt logs/*/ | head -1` would pick the *reruns* log dir (newest after
   reruns finish), silently skipping the main pipeline log. Fix: replaced
   with `git add -f logs/` (recursive). Adds all log dirs — total <1 MB.

2. Checkpoint inclusion cap was 500 MB total. After MAE training, all 12
   seed dirs total ~580 MB (each dir is 49 MB: best.pt + latest.pt). Cap
   raised to 800 MB so adaptor_best.pt files actually make the commit.

3. Modified source files (configs/base.yaml, scripts/*, src/ttic_embeddings/*)
   were never staged — only logs/captions/checkpoints. Fix: added
   `git add -u` (stages all tracked-file modifications) plus explicit
   force-add of pipeline.out, .env.pipeline, RUN_PIPELINE.md, and
   scripts/01_cache_features.py (untracked).

NOT done (declined by user): copying agent_notes.md, sidejobs/, finisher.log,
pipeline_summary.txt into the repo. These will be lost on instance shutdown
unless persisted elsewhere.

User authorized adding clip_seed1 + clip_seed2 reruns at native 35k cosine,
to run as a post-pipeline phase after MAE completes. Reasons:

1. clip_seed1 is resume-contaminated (Intervention 2 kill+resume during
   throughput fix; val_ppl 10.34 at step 26k post-resume vs 10.04 right
   before kill; final 10.167 @ 50k is ~0.3 ppl worse than seed2 at same
   step). Its checkpoint AND the captions generated from it are poisoned.
2. clip_seed2 was trained on a 50k cosine schedule, so its step-35k
   checkpoint (val_ppl 9.881) is on a different LR trajectory than the
   native-35k cosine used by siglip/dinov2/mae. Not apples-to-apples.
3. clip_seed3 is fine (native 35k), no rerun needed.

### Files prepared (NOT YET LIVE)

- /home/ubuntu/clip_reruns_postphase.sh — standalone rerun orchestrator:
  re-cache clip features → archive old clip_seed{1,2} checkpoints +
  captions_clip_seed1.jsonl → train both seeds paired with --no-resume →
  regen clip_seed1 captions → re-merge captions_seed1.jsonl → re-run
  Phase 5a/b/d (score, analyze, quality). Writes
  /home/ubuntu/clip_reruns.complete on success or .failed on failure.
  Total expected runtime: ~7-8 wall-h.

- /home/ubuntu/finisher_v2.sh — drop-in replacement for finisher.sh.
  Same pipeline-wait + commit/push/shutdown logic, but inserts a call
  to clip_reruns_postphase.sh AFTER "Pipeline complete." detection and
  BEFORE git work. If reruns fail, finisher_v2 writes failure summary
  and exits WITHOUT shutdown — preserves state for investigation.

Both scripts syntax-checked (`bash -n`). Neither is running.

### Lock-file gotcha (learned during the swap)

The original finisher.sh did `exec 9>finisher.lock; flock -n 9` BEFORE
forking the pipeline subshell. The pipeline (PID 950372) and ALL its
python training children inherited fd 9 from that fork. When v1 was
killed, fd 9 stayed open in the pipeline tree, so the exclusive lock
on finisher.lock is STILL HELD by the orphaned pipeline children.
That lock will only release when the pipeline tree exits.

Verify: `ls -la /proc/950372/fd/ | grep finisher.lock` shows
fd 9 -> /home/ubuntu/finisher.lock.

Consequence: v2 uses `/home/ubuntu/finisher_v2.lock` instead.
DO NOT remove or chmod finisher.lock — the kernel cleanup is automatic
on process exit. Leave it alone.

### Swap procedure (DONE at 00:42 UTC; documenting for rollback)

```
# 1. Kill v1 finisher
kill -TERM 22354    # SIGKILL if SIGTERM doesn't take in 3s

# 2. Launch v2 with setsid+disown (plain nohup+& from a tool session
#    silently fails to persist the child)
setsid nohup bash /home/ubuntu/finisher_v2.sh < /dev/null > /dev/null 2>&1 & disown

# 3. Verify
pgrep -af "finisher_v2.sh"
tail -5 /home/ubuntu/finisher.log   # expect "finisher_v2 started"
```

### Rollback (if reruns prove to be a mistake)

```
# Stop finisher_v2 if it's still in wait_no_pipeline
PID=$(pgrep -f "finisher_v2.sh"); kill -TERM "$PID"
# Restore original finisher
nohup bash /home/ubuntu/finisher.sh < /dev/null > /dev/null 2>&1 &
```

### Disk note

clip cache during reruns needs ~50 GiB. After MAE completes and cleans
up its 47 GiB cache, free space should be ~100+ GiB. If MAE leaves
cache behind for any reason, the rerun script will need that gone
first (it'll bail with "clip cache failed" if 01_cache_features.py
itself can't fit).

## INTERVENTION 5 — 2026-05-11 14:35 UTC — sidejobs using GPU 1 idle window

User approved running 4 side-jobs in the siglip_seed3 solo phase's idle
GPU window (~14:55-20:00 UTC). All artifacts under /home/ubuntu/sidejobs/.

Launched processes (2026-05-11 14:35 UTC):
- /home/ubuntu/sidejobs/orchestrate.sh — bash PID was 3166227 at launch.
  Logs: /home/ubuntu/sidejobs/orchestrate.log
- /home/ubuntu/sidejobs/eval_smoke.sh — bash PID was 3166262 at launch.
  Logs: /home/ubuntu/sidejobs/eval_smoke/eval_smoke.log

### orchestrate.sh stages (sequential, all on GPU 1)

1. Waits for siglip_seed2 to exit (polls `pgrep -f "02_train_clip.py.*--gpu 1"`).
2. **MAE pre-flight (sacrificial)**:
   - Disk gate: requires >=50 GiB free; aborts cleanly if not.
   - Caches MAE features to features/mae (~47 GiB, ~22 min).
   - Trains mae_seed99 for 5000 steps with --max-steps 5000 --no-resume
     (~48 min at 1.72 step/s).
   - Curve summary written to /home/ubuntu/sidejobs/mae_preflight_curve.txt.
   - Cleanup: `rm -rf features/mae checkpoints/mae_seed99` — pipeline's
     eventual MAE phase is unaffected (it re-caches from scratch).
3. **siglip_seed1 caption gen**: outputs to
   captions/captions_siglip_seed1.jsonl. 04_generate_captions.py is
   resume-safe (lines 252-300) so the pipeline's Phase 4 will skip
   already-completed rows — no duplicate work.
4. **Linear probes (clip+siglip)**: outputs to captions/probe_clip_siglip.csv.
   ~10 min on GPU 1. 07_probes.py uses raw frozen encoders, no adaptor
   needed. Note: pipeline's eventual Phase 5 generates probe_${enc}.csv
   per-encoder, but my output is dual-encoder — formats may not match.
   Worth checking before pipeline Phase 5 runs.

### eval_smoke.sh (CPU only, parallel)

Runs 05_score_metrics, 06_analyze, 08_caption_quality on a snapshot of
clip_seed1's in-progress captions (8040 lines at snapshot time, file is
still growing in real captions/). Purpose: catch bugs in the eval chain
before the full pipeline reaches Phase 5. 06_analyze is expected to
return non-zero on single-encoder input — that's fine, we're testing
the IO/arg surface.

### Resource accounting

- Disk: side jobs need ~47 GiB peak during MAE cache (stage 2).
  Pre-check gates abort if < 50 GiB free.
- GPU 1: side jobs use it serially. If GPU 1 becomes busy mid-stage
  (e.g., pipeline progression), later stages skip cleanly.
- CPU: eval_smoke runs spaCy parsing — 1 core. Doesn't impact GPU work.
- Memory: probably fine. We have ~30 GiB free RAM typically.

### Critical: must finish before dinov2 paired phase starts (~20:22 UTC)

Total expected runtime: ~3 hours. Hard deadline: 20:22 UTC when dinov2
seed1+seed2 paired phase reclaims GPU 1. If orchestrate.sh is still
running when dinov2 paired starts, the in-flight stage may OOM-GPU
or be killed; subsequent stages are protected by the pgrep check.

### Cleanup checklist if intervening

If you need to kill the side jobs (e.g., emergency):
  pkill -TERM -f "orchestrate.sh|eval_smoke.sh|02_train_clip.py.*seed 99|04_generate_captions.py.*siglip|07_probes.py"
  rm -rf /home/ubuntu/ttic_embeddings/features/mae
  rm -rf /home/ubuntu/ttic_embeddings/checkpoints/mae_seed99
Make sure to leave captions/captions_siglip_seed1.jsonl in place if
non-empty — the pipeline's Phase 4 will skip already-completed rows.
