#!/usr/bin/env bash
set -e

ENV_NAME="llm-lab2"
PYTHON_VERSION="3.11"

echo ">>> Creating conda env: $ENV_NAME (Python $PYTHON_VERSION)"

conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

echo ">>> Activating env (run this manually in your shell afterwards):"
echo "    conda activate $ENV_NAME"

echo ">>> Installing PyTorch (do this AFTER you activate env)"
echo "    # Gå till https://pytorch.org/get-started/locally/ och kör rätt kommando, t.ex:"
echo "    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"

echo ">>> Installing Python packages from requirements.txt"
echo "    pip install -r requirements.txt"

echo
echo "Done. Steps after this:"
echo "  1. conda activate $ENV_NAME"
echo "  2. Installera PyTorch med rätt CUDA enligt PyTorch-sidan"
echo "  3. pip install -r requirements.txt"
