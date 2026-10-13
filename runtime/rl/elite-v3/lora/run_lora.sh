#!/usr/bin/env bash
# elite-v3 commander LoRA (development evidence, train seeds only): two seed folds x (real labels, permuted-label
# control), hyperparameters fixed in REPORT.md before any result (1 epoch, lr 1e-4, rank 16, accumulate 8, seed 0).
# Runs in WSL on the RTX 5090 only (the trainer pins the GPU UUID and refuses anything else). Start it only while no
# native game job runs (CUDA init in WSL segfaults while ~32 game processes hold GPU contexts).
set -u
BASE=/mnt/c/Users/<user>/Documents/AlphaRush/runtime/rl/elite-v3/lora
PY=/home/<user>/alpharush/.venv/bin/python
TRAINER=$BASE/train_commander_lora.py
mkdir -p "$BASE/adapters" "$BASE/preds" "$BASE/logs"
for NAME in "$@"; do
  DATA=$BASE/data/$NAME
  for ATTEMPT in 1 2; do
    echo "$(date +%T) start $NAME attempt $ATTEMPT" >> "$BASE/logs/driver.log"
    PYTHONFAULTHANDLER=1 PYTHONDONTWRITEBYTECODE=1 "$PY" "$TRAINER" --train "$DATA/train.jsonl" \
      --holdout "$DATA/holdout.jsonl" --out "$BASE/adapters/$NAME" --predictions "$BASE/preds/$NAME.jsonl" \
      --epochs 1 --lr 1e-4 --rank 16 --accumulate 8 --eval-limit 1 --seed 0 > "$BASE/logs/$NAME.log" 2>&1
    CODE=$?
    echo "$(date +%T) end $NAME attempt $ATTEMPT exit $CODE" >> "$BASE/logs/driver.log"
    [ "$CODE" -eq 0 ] && break
  done
done
