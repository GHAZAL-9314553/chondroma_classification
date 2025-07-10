#!/bin/bash
#Set job requirements
#SBATCH -J Heidelberg-Preprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --mem=119G
#SBATCH --time=08:00:00

exec > "/projects/prjs1420/job_logs/${SLURM_JOB_ID}_combined.log" 2>&1
echo "Job started in $(pwd)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Home directory is $HOME"

#module load 2023
#module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
#module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
#module load matplotlib/3.7.2-gfbf-2023a
#module load scikit-learn/1.3.1-gfbf-2023a
#module load SciPy-bundle/2023.07-gfbf-2023a
#module load tqdm/4.66.1-GCCcore-12.3.0
#echo "Loaded required modules"

source /projects/prjs1420/venvgpu/bin/activate
echo "Activated virtual environment"

# Get numpy version
numpy_version=$(python -c "import numpy; print(numpy.__version__)")
echo "Numpy version: $numpy_version"

# Get pandas version
pandas_version=$(python -c "import pandas; print(pandas.__version__)")
echo "Pandas version: $pandas_version"

# Get the path of the pip executable
pip_path=$(which pip)
echo "Pip path: $pip_path"

pytorch_version=$(python -c "import torch; print(torch.__version__)")
echo "PyTorch version: $pytorch_version"

cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
echo "CUDA Available: $cuda_available"

# 2. Navigate to the code folder
cd /gpfs/work1/0/prjs1420/
echo "Changed to project directory: $(pwd)"

# 3. Set PYTHONPATH to current project root (for relative imports)
export PYTHONPATH=$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# 4. Run the CLI script with your config file
python /gpfs/work1/0/prjs1420/final_MIL.py 
sleep 1
echo "Finished job"
