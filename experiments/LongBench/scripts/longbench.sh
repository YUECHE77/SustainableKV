#!/bin/bash

# MODEL="mistral-7B-instruct-v0.2"
MODEL="longchat-v1.5-7b-32k"
# MODEL="lwm-text-chat-1m"
# MODEL="llama-2-7B-32k-instruct"

# COMPRESS_ARGS="ablation_c4096_w32_k7_maxpool.json"  # SnapKV
COMPRESS_ARGS="c4096_w32_sl9216_sink4_re10_k7_maxpool_pivot.json"  # SustainableKV

datasets=(
  # "narrativeqa"
  # "qasper"
  # "multifieldqa_en"
  "hotpotqa"
  "2wikimqa"
  "musique"
  "gov_report"
  "qmsum"
  "multi_news"
  "trec"
  "triviaqa"
  "samsum"
  "passage_count"
  "passage_retrieval_en"
  "lcc"
  "repobench-p"
)

for dataset in "${datasets[@]}"
do
  echo "Running dataset: $dataset"
  # python pred_snap.py --model "$MODEL" --compress_args_path "$COMPRESS_ARGS" --dataset "$dataset"  # SnapKV
  python pred_sustainable.py --model "$MODEL" --compress-args-path "$COMPRESS_ARGS" --dataset "$dataset"  # SustainableKV

  # Baseline:
  # python pred_snap.py --model "$MODEL" --dataset "$dataset"
done
