#!/bin/bash
#SBATCH -J UNI-FeatureExtraction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --mem=119G
#SBATCH --time=22:00:00

mkdir -p /projects/prjs1420/job_logs/
exec > "/projects/prjs1420/job_logs/${SLURM_JOB_ID}_combined.log" 2>&1

echo "í ½í´§ Job started in $(pwd)"
echo "í ¼í¶” Job ID: ${SLURM_JOB_ID}"
echo "í ¼í¿  Home directory: $HOME"

# ÙØ¹Ø§Ù„â€ŒØ³Ø§Ø²ÛŒ Ù…Ø­ÛŒØ· Ù…Ø¬Ø§Ø²ÛŒ
source /projects/prjs1420/venvgpu/bin/activate
echo "âœ… Activated virtual environment"

# Ù†Ø³Ø®Ù‡â€ŒÙ‡Ø§
echo "í ½í³¦ Versions:"
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import numpy; print('Numpy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# í ½í´ Ø±ÙØªÙ† Ø¨Ù‡ Ù…Ø³ÛŒØ± Ø¯Ø±Ø³Øª Ùˆ ØªÙ†Ø¸ÛŒÙ… PYTHONPATH
cd /gpfs/work1/0/prjs1420/UNI
export PYTHONPATH=$(pwd)
echo "í ½í³‚ Changed to project directory: $(pwd)"
echo "í ½í³ PYTHONPATH set to: $PYTHONPATH"

# í ½íº€ Ø§Ø¬Ø±Ø§ÛŒ Ø§Ø³ØªØ®Ø±Ø§Ø¬ feature Ø¨Ø§ Ù…Ø¯Ù„ UNI
python feature_extraction.py \
  --data_root /gpfs/work1/0/prjs1420/patches \
  --output_root /gpfs/work1/0/prjs1420/features/all \


echo "âœ… Finished feature extraction"
