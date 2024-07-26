set -e 
MODEL_PATH_ARG=$@
if [[ "$MODEL_PATH_ARG" == *"pythia2.8b"* ]]; then
    PEFT_ARG=" --base_model_name mnoukhov/pythia2.8b-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"pythia1b"* ]] && [[ "$MODEL_PATH_ARG" == *"lora"* ]]; then
    PEFT_ARG=" --base_model_name mnoukhov/pythia1b-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"pythia410m"* ]] && [[ "$MODEL_PATH_ARG" == *"lora"* ]]; then
    PEFT_ARG=" --base_model_name mnoukhov/pythia410m-sft-tldr"
else
    PEFT_ARG=""
fi
echo $PEFT_ARG

GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
if [[ "$GPU_MEMORY" == "16"* ]]; then
    # lazy check if we're using 16gb gpus
    BATCH_SIZE_ARG="--generate_batch_size 8"
else
    BATCH_SIZE_ARG=""
fi

TOKENIZERS_PARALLELISM=false /home/toolkit/.conda/envs/vllm0.4.0/bin/python generate_for_eval.py --config configs/generate_tldr.yml $MODEL_PATH_ARG $PEFT_ARG $BATCH_SIZE_ARG

if [[ "$MODEL_PATH_ARG" == *"pythia410m"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia410m-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"pythia1b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia1b-sft-tldr"
elif [[ "$MODEL_PATH_ARG" == *"pythia2.8b"* ]]; then
    REF_ARG=" --ref_model_name mnoukhov/pythia2.8b-sft-tldr"
else
    echo "output path doesn't contain one of model names"
    exit 1
fi

if [[ "$GPU_MEMORY" == "16"* ]]; then
    # lazy check if we're using 16gb gpus
    BATCH_SIZE_ARG="--eval_batch_size 4"
else
    BATCH_SIZE_ARG=""
fi

echo $BATCH_SIZE_ARG

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=$NPROC \
    load_and_eval.py --config configs/evaluate_tldr.yml $MODEL_PATH_ARG $REF_ARG
