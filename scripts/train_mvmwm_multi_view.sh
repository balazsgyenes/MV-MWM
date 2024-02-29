TASK=$1
USE_ROTATION=False
GPU=$3
SEED=$4

DISPLAY=:0.${GPU} TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python mvmwm/train.py \
--logdir ./logs/${TASK}/mvmwm_mv/$(date '+%Y-%m-%d/%H-%M-%S')/${SEED} \
--task ${TASK} \
--mae.view_masking 1 \
--mae.viewpoint_pos_emb True \
--use_rotation ${USE_ROTATION} \
--seed ${SEED}
