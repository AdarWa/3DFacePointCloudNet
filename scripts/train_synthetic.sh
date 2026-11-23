#!/usr/bin/env bash
# End-to-end pipeline: BFM2017 -> synthetic .npy faces -> classification
# pre-training -> (optional) triplet fine-tuning on real scans.
#
# Usage (from the repository root):
#   bash scripts/train_synthetic.sh
#   SUBJECTS=500 SCANS=20 EPOCHS=10 bash scripts/train_synthetic.sh
#   REAL_DATA=/path/to/real_npy bash scripts/train_synthetic.sh   # + fine-tune
#   bash scripts/train_synthetic.sh -resume                        # extra args go to train_cls
#
# Every stage is resumable: re-running skips subjects that are already
# generated, and -resume continues classification training.
set -euo pipefail
cd "$(dirname "$0")/.."

BFM=${BFM:-Data/model2017-1_face12_nomouth.h5}   # BFM2017 h5 model
OUT=${OUT:-Data/synthetic}                       # generated faces
RUN=${RUN:-runs/synthetic}                       # checkpoints + logs
SUBJECTS=${SUBJECTS:-1000}                       # identities (= classes)
SCANS=${SCANS:-20}                               # scans (expressions) per identity
POINTS=${POINTS:-4096}                           # points per scan
NOISE=${NOISE:-0.0}                              # vertex noise in mm
WORKERS=${WORKERS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
EPOCHS=${EPOCHS:-10}
BATCH=${BATCH:-32}
LR=${LR:-1e-3}
CLASSIFIER=${CLASSIFIER:-AL}                     # AL | MCP | L
REAL_DATA=${REAL_DATA:-}                         # set to fine-tune with train_triplet
TRIPLET_EPOCHS=${TRIPLET_EPOCHS:-20}

echo "== [1/3] Generating ${SUBJECTS} x ${SCANS} synthetic faces into ${OUT}"
python -m pointnet2.data.generate_faces \
    --bfm "$BFM" --out "$OUT" \
    --num_subjects "$SUBJECTS" --scans_per_subject "$SCANS" \
    --num_points "$POINTS" --noise_std "$NOISE" --workers "$WORKERS" --neutral_first

echo "== [2/3] Classification pre-training (${EPOCHS} epochs) -> ${RUN}"
python -m pointnet2.train.train_cls \
    -data_root "$OUT" -num_classes "$SUBJECTS" -num_points "$POINTS" \
    -epochs "$EPOCHS" -batch_size "$BATCH" -lr "$LR" --classifier_type "$CLASSIFIER" \
    -checkpoint_dir "$RUN" -log_file "$RUN/train_cls_log.txt" "$@"

if [[ -n "$REAL_DATA" ]]; then
    echo "== [3/3] Triplet fine-tuning on ${REAL_DATA}"
    python -m pointnet2.train.train_triplet \
        -data_root "$REAL_DATA" -model_checkpoint "$RUN/pointnet2_model_best.pth.tar" \
        -num_points "$POINTS" -epochs "$TRIPLET_EPOCHS" -augment \
        -checkpoint_dir "$RUN" -log_file "$RUN/triplet_log.txt"
else
    echo "== [3/3] Skipped triplet fine-tuning (set REAL_DATA=/path/to/real_npy to enable)"
fi
echo "Pre-trained backbone: $RUN/pointnet2_model_best.pth.tar"
