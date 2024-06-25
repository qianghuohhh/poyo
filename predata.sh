#!/bin/bash
#SBATCH -J ndt2_lu_try2
#SBATCH -o slurm_logs/%j.out
#SBATCH -p q_cn
#SBATCH -n 1

export http_proxy=10.11.100.5:3128 
export HTTP_PROXY=10.11.100.5:3128
export https_proxy=10.11.100.5:3128
export HTTPS_PROXY=10.11.100.5:3128
export ftp_proxy=10.11.100.5:3128
export FTP_PROXY=10.11.100.5:3128
export all_proxy=10.11.100.5:3128
export ALL_PROXY=10.11.100.5:3128

#source /home/yuezhifeng_lab/aochuan/DATA/bin/init_conda.sh
#conda activate onnx
source activate onnx
#wandb login eb9f8bffb20b4fbb7550ae857caad45673e6ebb8
#wandb login 88187bdc288b3a5ed707e99ac8daf5f72061f4c7
python -u /GPFS/yuezhifeng_lab_permanent/lutong/poyo/data/scripts/perich_miller_population_2018/prepare_data.py $@
