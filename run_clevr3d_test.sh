#!/bin/bash
#SBATCH --job-name=clevr3d_all
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=clevr3d_test_%j.out
#SBATCH --error=clevr3d_test_%j.err

echo "=== CLEVR 3D - All Gestalt Principles ==="
echo "Node: $(hostname)"
echo "Date: $(date)"

BLENDER_SIF="/mnt/vast/home/hs40vahe/GestaltReasoningMachines/blender.sif"
WORKDIR="/mnt/vast/home/hs40vahe/GestaltReasoningMachines/ELVIS"

cd "$WORKDIR"
apptainer exec --bind /mnt "$BLENDER_SIF" \
    env BLENDER_PATH=blender PYTHONUNBUFFERED=1 \
    python3 -u -m scripts.clevr3d.test_all_principles

echo ""
echo "--- Output structure ---"
find gen_data_3d/test_renders -name "*.png" | sort | head -60

echo ""
echo "=== Done ==="
echo "Date: $(date)"
