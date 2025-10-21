#!/bin/bash

CKPT="07b-pruned"

python -m llava.eval.model_vqa \
    --model-path /code/yingqi/models/liuhaotian/llava-v1.5-7b \
    --question-file /code/yingqi/LLaVA_dated/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /code/yingqi/LLaVA_dated/playground/data/eval/mm-vet/images \
    --answers-file /code/yingqi/LLaVA_dated/playground/data/eval/mm-vet/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /code/yingqi/LLaVA_dated/playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src /code/yingqi/LLaVA_dated/playground/data/eval/mm-vet/answers/${CKPT}.jsonl \
    --dst /code/yingqi/LLaVA_dated/playground/data/eval/mm-vet/results/${CKPT}.json

