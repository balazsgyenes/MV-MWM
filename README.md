# Multi-View Masked World Models for Visual Robotic Manipulation

Implementation of [MV-MWM](https://arxiv.org/abs/2302.02408) in TensorFlow 2.

## Method
Multi-View Masked World Models (MV-MWM) is a reinforcement learning framework that (i) trains a multi-view masked autoencoder with view-masking and (ii) learns a world model for single-view, multi-view, and viewpoint-robust control.

![MV-MWM Overview](https://user-images.githubusercontent.com/20944657/217286929-23c4bf7b-17e0-498a-b4b0-ace8d08fe118.gif)

## Instructions

Create and activate mamba environment:
```
mamba env create -p .env -f env.yaml
mamba activate ./.env
```

Install pc_rl (for maniskill2 environments):
```
pip install -e PATH/TO/PC_RL
```

There is no need to download the assets again, if they are already downloaded in pc_rl.

## Experiments

To reproduce our experiments, please run below scripts in `mvmwm` directory.

### Multi-View Control
```
source ./scripts/train_mvmwm_multi_view.sh {TASK} {USE_ROTATION} {GPU} {SEED}
# For instance,
source ./scripts/train_mvmwm_multi_view.sh rlbench_phone_on_base False 0 1
source ./scripts/train_mvmwm_multi_view.sh rlbench_stack_wine True 0 1
```

### Single-View Control
```
source ./scripts/train_mvmwm_single_view.sh {TASK} {USE_ROTATION} {GPU} {SEED}
# For instance,
source ./scripts/train_mvmwm_single_view.sh rlbench_phone_on_base False 0 1
source ./scripts/train_mvmwm_single_view.sh rlbench_stack_wine True 0 1
```

### Viewpoint-Robust Control
```
source ./scripts/train_mvmwm_viewpoint_robust.sh {TASK} {USE_ROTATION} {DIFFICULTY} {GPU} {SEED}
# For instance,
source ./scripts/train_mvmwm_viewpoint_robust.sh rlbench_phone_on_base_custom False medium 0 1
source ./scripts/train_mvmwm_viewpoint_robust.sh rlbench_stack_wine_custom True weak 0 1
```

## Note
This code might not perfectly reproduce the results in the paper, possible due to the human errors in preparing and cleaning the code for release. Please let us know if you have any problem or trouble in reproducing our results. We will also try to conduct sanity-check experiments as soon as possible.
