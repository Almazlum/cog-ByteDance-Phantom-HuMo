#!/bin/bash

mkdir -p ./output

python main.py humo/configs/inference/generate_single_gpu.yaml \
    generation.mode=TA \
    generation.positive_prompt=./examples/test_case.json \
    generation.output.dir=./output
