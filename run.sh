export CUDA_VISIBLE_DEVICES='3'
vllm serve /ssd/models/LLM-Research/Llama-2-7b-chat-hf \
    --host 0.0.0.0  \
    --port 8865 \
    --served-model-name llama8b \
    --tensor-parallel-size  1 \
    --max-model-len 128 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.90