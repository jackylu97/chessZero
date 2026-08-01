#!/bin/bash
# 2026-07-15: restart of 2026_07_08_prod_xl_ft_s800 with REANALYZE ON.
# 2026-07-17: mixture breakpoint 0.40 -> 0.12 (per-channel CE: anchor solved
#   0.08, warmstart flat 0.63, selfplay highest 0.95 & most starved). Adopts the
#   schedule's own 20/10/70 phase ~170k steps early. NOT removing warmstart:
#   graduation test (ckpt 72000 @800 sims vs SF-d8, the label engine) scored
#   0/1/23 = -669 Elo — the teacher is still far stronger; re-test each ~30-50k
#   steps and cut warmstart to 10%/0% only when score reaches ~35-50%.
# 2026-07-20: --resign-draws-only (resignation_relabel_policy_2026_07_20.md):
#   resignation no longer overwrites decisive outcomes — true losses keep their
#   conversion tails, comeback wins keep the win; only oracle-free draws flip,
#   with a TB-drawn-final veto. Takes effect at the NEXT restart.
# 2026-07-25: RESTORE warmstart share to 40/15/45 (was 20/10/70) from the 0.12
#   breakpoint onward, and HOLD it (removed the premature 0.60 taper to 10/10/80).
#   Reverses the 73k cut that caused the opening-value erosion (114k phase attrib:
#   deficit created move 6-10, monotone bleed to -3 pawns by move 16; value probe:
#   own-opening sign-acc 0.82->0.48, mean V on losing openings -0.24->+0.08).
#   Warmstart is the ONLY opening teacher (self-play openings are random/masked).
#   Do NOT re-taper warmstart until the graduation test says the model is near the
#   teacher. Gate: re-run phase attribution at ~125k — if eval-by-ply flattens,
#   openings recovering; if flat/worse, damage may be baked in (then reconsider
#   an earlier-checkpoint restart).
# 2026-07-24: drawn-seed archive (endgame_seeds_train_v2.txt, ~13% wdl=0 draw
#   seeds + wins; was 100% wdl=2). Repairs the TB-drawn value-calibration
#   regression (value probe 2026-07-23: |V|>0.5 on drawn 5-man 5/55->13/55 as
#   drawn positions starved from the diet — adjudication/resignation/decisive
#   seeds prune them). draw_frac 0.15; conversion metrics compute over wdl=2
#   seeds only, so seed/conversion_rate stays comparable; seed/draw_seed_count
#   & draw_seed_held now populate (defender-hold / attacker-false-win series).
#   ALSO: h2h series watcher must NOT run GPU matches concurrent with the
#   trainer — it OOM-crashed the run at step 111870 (reanalyze left the card
#   near-full, %3000 h2h match consumed the headroom). Series runs on CPU or
#   in a paused-trainer window only.
# 2026-07-22: veto/target-consistency bundle (veto_target_fix_2026_07_22.md),
#   CODE-SIDE ONLY (no flag changes): tensor engine now backs up 0.0 for pinned
#   draw children and pins pi'/A_{n+1} (was selection-only — stored policy
#   targets kept phantom-win mass on drawing moves); reanalyze passes the same
#   veto (board replay) and preserves the opening-mix loss mask; make_target
#   outcome parity is start_fen-aware; legacy anchor POV migrated on load.
#   Expect on restart: "Loaded replay buffer (N games)" then anchor injections
#   log normally; reanalyze log line gains "opening plies preserved".
# Identical to the original _ft launch except:
#   --resume            -> newest checkpoint that has a matching .buf
#   --reanalyze-interval 1130   (was 0; 1:1 with self-play interval)
#   --reanalyze-batch-size 1024 (~1x buffer-lifetime refresh per game)
#   --reanalyze-sims 800        (2026-07-15b: bumped 128->800 per user — full
#                                self-play search depth so reanalyzed targets are
#                                strictly comparable to originals; ~88 min/call est.
#                                First call at 128 sims (step 67800) remains in buffer.)
# Context: reanalyze+gumbel previously hard-failed on the tensor path
# (trainer.py NotImplementedError); now routes run_batch_gpu. Parity vs numpy
# verified equivalent-within-search-noise; tests green.
set -euo pipefail
cd /home/user/chessZero

RUN_DIR=checkpoints/chess/2026_07_08_prod_xl_ft_s800
# newest step with BOTH .pt and .buf (resume without .buf would re-bootstrap the buffer)
# 2026-07-22: sort NUMERICALLY on the step extracted from the filename. The old
# `sort -t_ -k2 -rn` keyed on a path component (the run id is full of "_"),
# degrading to lexicographic order — first restart past step 100000 resumed
# from 90000 ("9..." > "104000" lexicographically). Caught at relaunch.
# RESUME_OVERRIDE lets a caller pin a specific checkpoint (e.g. to fork from an
# earlier point) instead of the newest-with-.buf. Used 2026-07-26 to restart the
# warmstart experiment from 110000 (the last warmstart-intact checkpoint before
# the injection-pool exhaustion collapsed batch_warmstart_frac to 0 at 111k).
RESUME="${RESUME_OVERRIDE:-}"
if [ -n "$RESUME" ] && [ ! -f "${RESUME%.pt}.buf" ]; then
  echo "FATAL: RESUME_OVERRIDE $RESUME has no matching .buf"; exit 1
fi
if [ -z "$RESUME" ]; then
  for pt in $(ls -1 "$RUN_DIR"/checkpoint_*.pt \
              | sed -E 's/.*checkpoint_([0-9]+)\.pt$/\1 &/' | sort -rn | cut -d' ' -f2); do
    buf="${pt%.pt}.buf"
    if [ -f "$buf" ]; then RESUME="$pt"; break; fi
  done
fi
if [ -z "$RESUME" ]; then echo "FATAL: no checkpoint with matching .buf"; exit 1; fi
echo "Resuming from: $RESUME (buf: $(du -h "${RESUME%.pt}.buf" | cut -f1))"

# 2026-07-25: torch.compile / Triton caches redirected off the cramped root fs
# (/tmp is on `/`, 49G, shared with other projects' data — it hit 100% and the
# reanalyze Triton compile crashed the trainer with OSError ENOSPC). /home has
# 57G free. Keeps the recompilable cache on the roomy volume.
TORCHINDUCTOR_CACHE_DIR=/home/user/chessZero/.cache/torchinductor \
TRITON_CACHE_DIR=/home/user/chessZero/.cache/triton \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=. \
  .venv/bin/python -u scripts/_faulthandler_bootstrap.py scripts/train.py \
  --game chess_hybrid_xl --run-id 2026_07_08_prod_xl_ft_s800 --device cuda \
  --resume "$RESUME" \
  --policy-head-type from_to \
  --reward-head-planes 8 \
  --use-gumbel --gumbel-m 16 --per-alpha 0 \
  --num-simulations 800 \
  --num-self-play-games 1024 --num-parallel-games 1024 \
  --self-play-interval 1130 \
  --replay-buffer-size 10240 \
  --steps 600000 --eval-interval 2000 --mask-illegal-policy \
  --self-play-warmup-frac 0.05 --warmstart-buffer-size 300 \
  --stockfish-injection-path data/stockfish_injection \
  --stockfish-injection-games 300 --stockfish-injection-interval 256 \
  --warmstart-sample-frac 0.4 --warmstart-sample-frac-final 0.1 --warmstart-anneal-frac 0.6 \
  --batch-mixture-schedule '[[0.00,{"warmstart":0.70,"anchor":0.30,"selfplay":0.00}],[0.05,{"warmstart":0.40,"anchor":0.15,"selfplay":0.45}],[0.12,{"warmstart":0.40,"anchor":0.15,"selfplay":0.45}],[0.60,{"warmstart":0.40,"anchor":0.15,"selfplay":0.45}]]' \
  --anchor-max-size 1024 \
  --merged-seed-batch \
  --opening-mix-mean-plies 6 \
  --seed-curriculum \
  --tb-value-hard \
  --tb-policy-weight 0.5 --tb-policy-weight-final 0.2 --tb-policy-anneal-frac 0.6 \
  --material-value-weight 0.5 --material-value-anneal-frac 0.6 \
  --use-material-head --material-head-loss-weight 0.25 \
  --root-terminal-draws --root-terminal-draws-min-repeats 2 \
  --resign-enabled --resign-holdout-frac 0.20 --resign-exempt-seeded \
  --resign-draws-only \
  --reanalyze-interval 1130 --reanalyze-batch-size 1024 --reanalyze-sims 800 \
  --tb-root-probe --tb-path data/syzygy --tb-gaviota-path data/gaviota \
  --tb-relabel-workers 8 \
  --tb-value-weight 1.0 --tb-value-dtz-shape 0.0 --tb-moves-left-weight 1.0 \
  --endgame-seed-frac 0.30 --endgame-seed-archive data/endgame_seeds_train_v2.txt \
  --tb-anchor-path data/tb_anchor --tb-anchor-games 64 --tb-anchor-interval 256 \
  --tb-rollout-fill \
  2>&1 | tee selfplay_production.log
