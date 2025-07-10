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

echo "�� Job started in $(pwd)"
echo "�� Job ID: ${SLURM_JOB_ID}"
echo "�� Home directory: $HOME"

# فعال‌سازی محیط مجازی
source /projects/prjs1420/venvgpu/bin/activate
echo "✅ Activated virtual environment"

# نسخه‌ها
echo "�� Versions:"
python -c "import torch; print('Torch:', torch.__version__)"
python -c "import numpy; print('Numpy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# �� رفتن به مسیر درست و تنظیم PYTHONPATH
cd /gpfs/work1/0/prjs1420/UNI
export PYTHONPATH=$(pwd)
echo "�� Changed to project directory: $(pwd)"
echo "�� PYTHONPATH set to: $PYTHONPATH"

# �� اجرای استخراج feature با مدل UNI
python feature_extraction.py \
  --data_root /gpfs/work1/0/prjs1420/patches \
  --output_root /gpfs/work1/0/prjs1420/features/all \


echo "✅ Finished feature extraction"
