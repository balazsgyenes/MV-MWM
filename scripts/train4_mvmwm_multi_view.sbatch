#!/bin/sh

#SBATCH --time=2880
#SBATCH --partition=accelerated
#SBATCH --gres=gpu:4
#SBATCH --error=/hkfs/work/workspace/scratch/jh8109-pc_rl/repos/MV-MWM/slurm/%j_0_log.err
#SBATCH --output=/hkfs/work/workspace/scratch/jh8109-pc_rl/repos/MV-MWM/slurm/%j_0_log.out
#SBATCH --open-mode=append
#SBATCH --job-name=train


TASK=$1
USE_ROTATION=False

GPU=0
SEED=42
DISPLAY=:0.${GPU} TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python mvmwm/train.py \
--logdir ./logs/${TASK}/mvmwm_mv/$(date '+%Y-%m-%d/%H-%M-%S')/${SEED} \
--task ${TASK} \
--mae.view_masking 1 \
--mae.viewpoint_pos_emb True \
--use_rotation ${USE_ROTATION} \
--seed ${SEED} &

sleep 5

GPU=1
SEED=43
DISPLAY=:0.${GPU} TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python mvmwm/train.py \
--logdir ./logs/${TASK}/mvmwm_mv/$(date '+%Y-%m-%d/%H-%M-%S')/${SEED} \
--task ${TASK} \
--mae.view_masking 1 \
--mae.viewpoint_pos_emb True \
--use_rotation ${USE_ROTATION} \
--seed ${SEED} &

sleep 5

GPU=2
SEED=44
DISPLAY=:0.${GPU} TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python mvmwm/train.py \
--logdir ./logs/${TASK}/mvmwm_mv/$(date '+%Y-%m-%d/%H-%M-%S')/${SEED} \
--task ${TASK} \
--mae.view_masking 1 \
--mae.viewpoint_pos_emb True \
--use_rotation ${USE_ROTATION} \
--seed ${SEED} &

sleep 5

GPU=3
SEED=45
DISPLAY=:0.${GPU} TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python mvmwm/train.py \
--logdir ./logs/${TASK}/mvmwm_mv/$(date '+%Y-%m-%d/%H-%M-%S')/${SEED} \
--task ${TASK} \
--mae.view_masking 1 \
--mae.viewpoint_pos_emb True \
--use_rotation ${USE_ROTATION} \
--seed ${SEED} &

wait
