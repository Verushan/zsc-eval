#!/bin/bash
# Stage-1 self-play arms for the MORL benchmark.
#
# Every arm runs the *same* train script, runner, architecture and
# hyper-parameters; only the reward the PPO buffer sees differs. That is the
# whole point -- it isolates "what reward was optimised" as the single
# independent variable.
#
#   bench_sp        sparse + reward_shaping_factor * hand-shaped   (ZSC-Eval baseline)
#   bench_sparse    sparse only, no shaping at all                 (lower-bound control)
#   bench_morl      w . r_vec, uniform fixed w                     (proposed)
#   bench_morl_ad   w . r_vec, mirror-descent adaptive w           (proposal 4.2.3)
#
# `--morl_objectives default` is passed to *every* arm, including the two that
# do not use the objective vector as their reward, so all four log an identical
# `ep_obj_*` behavioural breakdown and can be compared directly.
#
# bench_sparse is expressed through the MORL path with w = (20,0,0,0), which
# `morl/check_morl_reward.py::sp_equivalence` proves is bit-for-bit the sparse
# reward. Routing it through the same code path keeps the reward the only
# difference between it and bench_morl.
#
# Usage: bash shell/train_morl_benchmark.sh <layout> <arm> [seed_begin] [seed_max]
env="Overcooked"

layout=$1
arm=$2
seed_begin=${3:-1}
seed_max=${4:-3}

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    echo "MORL is only supported on the old-env layouts, got '${layout}'"
    exit 1
fi

# The objective vector, and the arm-specific reward wiring.
objectives="default"
case "${arm}" in
    bench_sp)
        morl_flags=(--morl_objectives ${objectives})
        ;;
    bench_sparse)
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "20,0,0,0")
        ;;
    bench_morl)
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "0.25,0.25,0.25,0.25")
        ;;
    bench_morl_ad)
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "0.25,0.25,0.25,0.25" --morl_adaptive_weights)
        ;;
    *)
        echo "Unknown arm '${arm}'. Expected one of bench_sp bench_sparse bench_morl bench_morl_ad"
        exit 1
        ;;
esac

# Short-run budget. Unlike train_sp.sh, the entropy and reward-shaping horizons
# are scaled to num_env_steps rather than left at the paper's 1e7: on a 2e6-step
# run the upstream "0 5e6 1e7" schedule never leaves the 0.2 entropy phase, so
# every arm would be measured mid-exploration.
#
# MORL_BENCH_STEPS overrides the budget and rescales both schedules with it, so a
# longer run still leaves the 0.2 entropy phase at the same fraction of training.
# It carries its own name for the reason every other override in this repo does:
# .env is sourced first, so a generic name there would silently win.
num_env_steps=${MORL_BENCH_STEPS:-2e6}
reward_shaping_horizon=${num_env_steps}
entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 $(awk -v n="${num_env_steps}" 'BEGIN{printf "%d", n / 2}') ${num_env_steps}"

episode_length=400
ppo_epoch=15
num_mini_batch=2
num_agents=2
algo="mappo"
# NOTE: `--use_recurrent_policy` is `action="store_false"` in zsceval/config.py,
# so passing it *disables* the RNN. That is deliberate and matches the existing
# random0 policy pool, whose checkpoints carry no rnn.* tensors and are loaded
# through mlp_policy_config.pkl. algorithm_name=mappo asserts on it either way.
ulimit -n 65536

echo "env ${env}, layout ${layout}, algo ${algo}, arm ${arm}, seeds ${seed_begin}..${seed_max}, steps ${num_env_steps}"
for seed in $(seq ${seed_begin} ${seed_max});
do
    echo "=== ${arm} seed ${seed} ==="
    python train/train_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${arm} \
    --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads $TRAINING_THREADS --n_rollout_threads $ROLLOUT_THREADS \
    --num_mini_batch ${num_mini_batch} --episode_length ${episode_length} \
    --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    "${morl_flags[@]}" \
    --ppo_epoch ${ppo_epoch} --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --cnn_layers_params "32,3,1,1 64,3,1,1 32,3,1,1" --use_recurrent_policy \
    --use_proper_time_limits \
    --save_interval 25 --log_interval 10 --use_eval --eval_interval 50 --eval_episodes 12 \
    --wandb_tags morl-benchmark ${arm} \
    --wandb_name $WANDB_ENTITY || exit 1
done
