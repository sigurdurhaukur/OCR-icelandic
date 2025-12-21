#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --job-name=gotocr2_icelandic_v1
#SBATCH --output=logs/gotocr2_icelandic_v1-%j.log
#SBATCH --error=logs/gotocr2_icelandic_v1-%j.err
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -e  # Exit on any error

SCRATCH=/scratch/s5982960/OCR-icelandic
# export HF_HOME=/tmp

module purge
module load CUDA/12.6.0
module load Python/3.13.1-GCCcore-14.2.0
module load Boost/1.79.0-GCC-11.3.0

mkdir -p $SCRATCH

# Experiment configuration
EXPERIMENT_NAME="gotocr2-icelandic-v1"
MODEL_VERSION="base"
DATASET_VERSION="Sigurdur/isl_synthetic_ocr"
LORA_CONFIG="r16-alpha32"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${EXPERIMENT_NAME}-${MODEL_VERSION}-${DATASET_VERSION}-${LORA_CONFIG}-${TIMESTAMP}"

# Log system info
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Run Name: $RUN_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "========================"

# Copy data to scratch if needed
# currently handled by the training script

# Copy data from scratch
# currently handled by the training script

# run our fine-tuning script
cd $SCRATCH
source $SCRATCH/.venv/bin/activate
uv sync

python3 scripts/smol_vlm_ft.py \
    push_to_hub=True \
    per_device_train_batch_size=4 \
    per_device_eval_batch_size=4 \
    logging_steps=50 \
    save_steps=500 \
    warmup_steps=500 \
    output_dir="GOT-OCR-2.0-hf" \
    model_id="stepfun-ai/GOT-OCR-2.0-hf" \
    hf_hub_output_model_id="Sigurdur/gotocr2-icelandic-v1" \
    eval_steps=125 \
    run_name="gotocr2_icelandic_v1" \
    learning_rate=3e-4 \
    hf_dataset_id="Sigurdur/isl_synthetic_ocr" \


# Log completion
echo "=== Job Completion ==="
echo "End Time: $(date)"
echo "======================="