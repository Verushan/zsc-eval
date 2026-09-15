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

# The objective layer exists in both env packages (see the MORL section of
# CLAUDE.md), so the multi-recipe layouts train the same arms as the old ones.
# This used to exit here, which meant the _m and _mx layouts had partners but
# could never have ego agents to evaluate against them.
if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

# The objective vector, and the arm-specific reward wiring.
# `default` is the original four-objective set. `anchored` swaps coordination
# for the version credited only once a handed object is used -- the plain one
# is farmable and six seeds found the loop, so new runs should use anchored.
objectives=${OBJECTIVES:-default}
# Uniform w over however many objectives the set has. MORL_WEIGHTS overrides it.
# The four-way literal was hardcoded, so a three-objective set would silently
# have been handed a fourth weight for a component that does not exist.
case "${objectives}" in
    anchored_live3) uniform_w="0.3333,0.3333,0.3333" ;;
    recipe)         uniform_w="0.1667,0.1667,0.1667,0.1667,0.1667,0.1667" ;;
    *)              uniform_w="0.25,0.25,0.25,0.25" ;;
esac
uniform_w=${MORL_WEIGHTS:-$uniform_w}
# Task at the environment's delivery value, dense objectives at 3 (the
# baseline's per-event shaping value), for the annealed arms.
case "${objectives}" in
    anchored_live3) ann_w="20,3,3" ;;
    recipe)         ann_w="20,3,3,3,3,3" ;;
    *)              ann_w="20,3,3,3" ;;
esac
ANN_WEIGHTS=${ANN_WEIGHTS:-$ann_w}
# MORL_TEAM_TASK=1 credits deliveries to the team, as the baseline does. The
# annealed arms need it: with per-agent credit the prepper is paid nothing at
# the end of training and the run converges at the sparse-only level.
team_flag=()
[ "${MORL_TEAM_TASK:-0}" = "1" ] && team_flag=(--morl_team_task)
# Appended to the W&B experiment_name only, never to the arm. Runs made under
# a different objective set must not share an experiment_name with the ones
# they replace: extract_sp_models filters on experiment_name, so a re-baseline
# under `anchored` would otherwise be silently mixed with the `default` runs it
# exists to supersede.
exp_suffix=${EXP_SUFFIX:-}
case "${arm}" in
    bench_sp)
        morl_flags=(--morl_objectives ${objectives})
        ;;
    bench_sparse)
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "20,0,0,0")
        ;;
    bench_morl)
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "${uniform_w}")
        ;;
    bench_morl_ad)
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "${uniform_w}" --morl_adaptive_weights)
        ;;
    bench_morl_ad_obs)
        # Adaptive w, with the live w appended to the observation.
        #
        # The mirror-descent update moves w mid-episode while the agent has no
        # way to see that it moved, so two identical observations carry
        # different returns -- the reward is non-Markovian and the agent is
        # being asked to adapt to a signal it cannot perceive.
        # --use_morl_obs_weights appends w as K constant channels and restores
        # the Markov property. It has never been used in a benchmark arm, so
        # every adaptive result so far is from the version that cannot see its
        # own preferences.
        #
        # It widens the observation, so these agents need their own policy
        # config (prep/store_policy_config.py --morl_obs_weights) and cannot
        # load the shared one. env_policy trims each frozen partner back to the
        # width it was built for.
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "${uniform_w}" --morl_adaptive_weights --use_morl_obs_weights)
        ;;
    bench_morl_ann)
        # Uniform-w MORL with the dense objectives annealed like the baseline's
        # shaping term: task 20 per delivery (the environment's own value),
        # every other objective 3 per event, and those 3s decay to 0 over the
        # shaping horizon. What the agent optimises at the end of training is
        # the task alone, which is the property the hand-shaped baseline had
        # and every MORL arm so far lacked (deliveries were 17% of the
        # converged MORL reward on unident_s).
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "${ANN_WEIGHTS}" --morl_anneal_dense "${team_flag[@]}")
        ;;
    bench_morl_fill)
        # The partner-conditioned agent: annealed dense objectives as above,
        # plus the state a partner-conditioned policy needs (own and partner
        # objective mix, --use_morl_obs_shares; own w, --use_morl_obs_weights)
        # and the complement rule, which steers each agent's w toward the
        # objectives its partner is doing least of. Widens the observation by
        # 3K channels; needs its own policy config to cross-play.
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "${ANN_WEIGHTS}" --morl_anneal_dense "${team_flag[@]}"
                    --morl_adaptive_weights --morl_adaptive_target complement
                    --use_morl_obs_weights --use_morl_obs_shares)
        ;;
    bench_sp_shares)
        # The baseline reward with the partner state. Isolates "does seeing
        # what the partner does help" from everything MORL.
        morl_flags=(--morl_objectives ${objectives} --use_morl_obs_shares)
        ;;
    bench_morl_div)
        # Weights are per-seed; set inside the loop below.
        morl_flags=()
        ;;
    *)
        echo "Unknown arm '${arm}'. Expected one of bench_sp bench_sparse bench_morl bench_morl_ad bench_morl_ad_obs bench_morl_ann bench_morl_fill bench_sp_shares bench_morl_div"
        exit 1
        ;;
esac

# bench_morl_div: one weight vector per population member.
#
# Every other MORL arm gives *every* seed the same w, so its population differs
# only by random initialisation -- exactly like bench_sp. That is not using a
# multi-objective reward to make a population diverse, it is using it to change
# what the whole population optimises. This arm is the other thing: member i
# gets its own w_i, so members differ in *what they are trying to do*.
#
# The design is deterministic rather than sampled, so a seed reproduces its
# vector, and it is laid out as the three vertices of the simplex over the
# non-task objectives followed by the three edge midpoints:
#
#   seed 1  prep          seed 4  prep + plating
#   seed 2  plating       seed 5  prep + coordination
#   seed 3  coordination  seed 6  plating + coordination
#
# task_completion is held at 0.25 in every member rather than being one of the
# spread dimensions. A member with no task weight has no delivery signal at all,
# and the failure mode that produces is already on record: random0's
# bench_sparse population has the highest behavioural diversity of any arm
# (0.70) and the lowest outcome diversity (0.00), because its members flail in
# different ways and none of them score. Diversity is only worth having among
# members that can actually cook.
#
# Every vector sums to 1.0, matching the L1 norm of bench_morl's uniform
# 0.25x4, so the arms differ in the *direction* of w and not in reward scale --
# which would otherwise act as a per-arm learning-rate change.
DIV_WEIGHTS=(
    "0.25,0.75,0,0"        # 1  ingredient_prep
    "0.25,0,0.75,0"        # 2  plating
    "0.25,0,0,0.75"        # 3  coordination
    "0.25,0.375,0.375,0"   # 4  prep + plating
    "0.25,0.375,0,0.375"   # 5  prep + coordination
    "0.25,0,0.375,0.375"   # 6  plating + coordination
)

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
    if [ "${arm}" = "bench_morl_div" ]; then
        # Seeds are 1-based; wrap so a seed range longer than the table still
        # runs, rather than silently training with an empty --morl_weights.
        w=${DIV_WEIGHTS[$(( (seed - 1) % ${#DIV_WEIGHTS[@]} ))]}
        morl_flags=(--use_morl --morl_objectives ${objectives} --morl_weights "${w}")
        echo "    w = ${w}"
    fi
    python train/train_sp.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${arm}${exp_suffix} \
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
    --wandb_tags morl-benchmark ${arm}${exp_suffix} ${objectives} \
    --wandb_name $WANDB_ENTITY || exit 1
done
