#!/bin/bash
#Instructions from: 
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/speechllm_fc_llama2_7b
set -x
MEGATRON_CKPT=/workspace_host/salm/speechllm_fc_llama2_7b.nemo
CUDA_VISIBLE_DEVICES=0 python export_to_onnx.py \
	model.restore_from_path=$MEGATRON_CKPT \
	model.export_to_path=speechllm_fc_llama2_7b

