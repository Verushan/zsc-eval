#!/bin/bash
# Stage 1 self-play, but the RL reward is the scalarized MORL objective vector
# (zsceval/envs/morl) instead of sparse + hand-crafted shaped reward.
#
# Identical to train_sp.sh in every other respect -- same train script, same
# runner, same hyper-parameters -- so `sp` and `morl` runs are directly
# comparable. Only zsceval/envs/overcooked/ implements the objective layer, so
# this is restricted to the "old" layouts.
env="Overcooked"

layout=$1
if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    echo "MORL is only supported on the old-env layouts, got '${layout}'"
    exit 1
fi

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 5e6 1e7"

if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 8e6 1e7"
fi

reward_shaping_horizon="1e6"
num_env_steps="1e6"
episode_length=400
ppo_epoch=15
num_mini_batch=2

# task_completion, ingredient_prep, plating, coordination
objectives="default"
weights="0.25,0.25,0.25,0.25"

num_agents=2
algo="mappo"
exp="morl"
seed_begin=2
seed_max=4
ulimit -n 65536

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp}, objectives are ${objectives}, seed from ${seed_begin} to ${seed_max}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "seed is ${seed}:"
    python train/train_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads $TRAINING_THREADS --n_rollout_threads $ROLLOUT_THREADS \
    --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} \
    --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --use_morl --morl_objectives ${objectives} --morl_weights ${weights} \
    --ppo_epoch ${ppo_epoch} --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 1000 --log_interval 10 --use_eval --eval_interval 250 \
    --wandb_name $WANDB_ENTITY
done
