#!/usr/bin/env bash
set -e

ENV_NAME="ai_lib-actor_critic_mountain_car"

# Enable conda inside shell script
source "$(conda info --base)/etc/profile.d/conda.sh"

# Make sure we are at repository root
cd "$(dirname "$0")/../../.."

# Create environment only if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "$ENV_NAME" python=3.10 -y

    conda activate "$ENV_NAME"

    pip install torch --index-url https://download.pytorch.org/whl/cu130
    pip install gymnasium

    pip install -e .
else
    conda activate "$ENV_NAME"
fi

python experiments/reinforcement_learning/actor_critic_mountain_car/main.py

conda deactivate