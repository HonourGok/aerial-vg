#!/usr/bin/env bash
# Train + evaluate one condition.
# usage: scripts/run_one.sh <name> <geo_aug> <gpu_id>
# env:   DEVICE (cuda), BERT (bert-base-uncased), BERT_LAYERS (12), WORKERS (4)
#        DATA_ROOT (folder with Annotations/ and JPEGImages/), DETR (path to detr-r50-e632da11.pth)
#        VAL_MAX (500, per-epoch validation subset), TEST_LIMIT (0 = full test), LAT (latency samples)
#        EPOCHS (7), LR_DROP (5), FRACTION (1.0), BS (8), IMSIZE (640), MAX_HOURS (5), SEED (13)
set -euo pipefail
NAME=$1; GEO=$2; GPU=${3:-0}
EPOCHS=${EPOCHS:-7}; LR_DROP=${LR_DROP:-5}; FRACTION=${FRACTION:-1.0}; BS=${BS:-8}
IMSIZE=${IMSIZE:-640}; MAX_HOURS=${MAX_HOURS:-5}; SEED=${SEED:-13}
VAL_MAX=${VAL_MAX:-500}; BERT=${BERT:-bert-base-uncased}; BERT_LAYERS=${BERT_LAYERS:-12}; DEVICE=${DEVICE:-cuda}; WORKERS=${WORKERS:-4}
TEST_LIMIT=${TEST_LIMIT:-0}; LAT=${LAT:-200}
AMP="--amp"; [ "$DEVICE" = "cpu" ] && AMP=""
OUT=outputs/$NAME
RESUME=""
if [ -f "$OUT/checkpoint.pth" ]; then RESUME="--resume $OUT/checkpoint.pth"; fi

CUDA_VISIBLE_DEVICES=$GPU python train.py --vl_prenorm 1 --warmup_iters 500 --clip_max_norm 0.1 \
  --dataset dior_rsvg --data_root "$DATA_ROOT" --split_root splits \
  --detr_model "$DETR" --bert_model "$BERT" --device $DEVICE \
  --backbone resnet50 --bert_enc_num $BERT_LAYERS --detr_enc_num 6 \
  --imsize $IMSIZE --max_query_len 40 --batch_size $BS \
  --lr 1e-4 --lr_bert 1e-5 --lr_visu_cnn 1e-5 --lr_visu_tra 1e-5 \
  --epochs $EPOCHS --lr_scheduler step --lr_drop $LR_DROP \
  --aug_crop --aug_scale --aug_translate \
  --geo_aug $GEO --train_fraction $FRACTION --seed $SEED --val_max $VAL_MAX \
  $AMP --num_workers $WORKERS --max_hours $MAX_HOURS \
  --output_dir $OUT $RESUME

CKPT=$OUT/best_checkpoint.pth; [ -f "$CKPT" ] || CKPT=$OUT/checkpoint.pth
CUDA_VISIBLE_DEVICES=$GPU python eval_dior.py --checkpoint $CKPT --data_root "$DATA_ROOT" \
  --split_root splits --out results/$NAME --device $DEVICE \
  --limit $TEST_LIMIT --latency_samples $LAT
