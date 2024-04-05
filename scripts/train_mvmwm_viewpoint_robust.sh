TASK=$1
USE_ROTATION=$2
DIFFICULTY=$3
GPU=$4
SEED=$5

DISPLAY=:0.${GPU} TF_XLA_FLAGS=--tf_xla_auto_jit=2 CUDA_VISIBLE_DEVICES=${GPU} \
python mvmwm/train.py \
--logdir ./logs/${TASK}/mvmwm_${DIFFICULTY}/${SEED} \
--configs multiview_${DIFFICULTY} eval_strong \
--use_randomize True \
--task ${TASK} \
--mae.view_masking 1 \
--mae.viewpoint_pos_emb False \
--use_rotation ${USE_ROTATION} \
--augment True \
--seed ${SEED}