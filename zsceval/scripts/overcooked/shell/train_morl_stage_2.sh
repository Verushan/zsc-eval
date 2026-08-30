#!/bin/bash
# Stage 2 of the ZSC-Eval pipeline, run once per MORL-benchmark arm.
#
# The stage-1 benchmark answered "what reward should a self-play agent
# optimise". This answers the question that one could not: what does the
# *pipeline* produce when the population it trains against was raised on that
# reward. FCP stage 2 trains a single recurrent best-response agent against a
# frozen population of stage-1 checkpoints; here the population is one arm's
# three seeds at three checkpoints each (9 partners), so the arms are matched on
# population size and differ only in the reward those partners optimised.
#
# The ego agent's own reward is the *task* reward (sparse + annealed shaping) for
# every arm, deliberately. Stage 2 is where an agent is supposed to learn to
# adapt to partners, not where the objective vector is under test; giving the
# MORL arms a MORL ego reward as well would confound "MORL population" with
# "MORL ego" and neither result would be readable. `--morl_objectives default`
# is still passed so the env is built identically to stage 1, but note that it
# buys no extra logging here: evaluate_with_multi_policy drops every key except
# eval_ep_sparse_r / eval_ep_shaped_r once stage == 2 and wandb is on. The
# stage-2 objective breakdown comes from the cross-play pass instead, which
# keeps all of them.
#
# The optional trailing suffix goes into the experiment name (fcp-S2-{arm}{sfx})
# and the pool directory. A reduced pilot run and the full run would otherwise
# share an experiment name, and extract_S2_models.py keys checkpoints by seed --
# so the second set would silently overwrite the first.
#
# Usage: bash shell/train_morl_stage_2.sh <layout> <arm> [seed_begin] [seed_max] [num_env_steps] [exp_suffix]
env="Overcooked"

layout=$1
arm=$2
seed_begin=${3:-1}
seed_max=${4:-3}
num_env_steps=${5:-2e6}
exp_suffix=${6:-""}

if [[ -z "${layout}" || -z "${arm}" ]]; then
    echo "usage: bash shell/train_morl_stage_2.sh <layout> <arm> [seed_begin] [seed_max] [num_env_steps]"
    exit 1
fi

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    echo "The benchmark arms only exist on the old-env layouts, got '${layout}'"
    exit 1
fi

yml="${POLICY_POOL}/${layout}/fcp/s2/train-${arm}.yml"
if [[ ! -f "${yml}" ]]; then
    echo "missing ${yml} -- run prep/gen_arm_S2_yml.py ${layout} --arm ${arm} first"
    exit 1
fi

# The runner asserts --population_size == len(population), and a mismatch is
# only discovered after the environments have spun up, so count the partners
# out of the yml rather than hardcoding 9.
population_size=$(grep -c "train: False" "${yml}")
if [[ ${population_size} -lt 1 ]]; then
    echo "no frozen partners in ${yml}"
    exit 1
fi

# Same lesson as stage 1: the upstream "0 4e7 8e7" entropy schedule is written
# for an 8e7-step run, so on a short one the coefficient never leaves its 0.2
# exploration phase and every arm is measured mid-exploration. Scale the
# schedule to the budget instead.
entropy_coefs="0.2 0.05 0.01"
entropy_mid=$(awk -v s="${num_env_steps}" 'BEGIN{printf "%.0f", s/2}')
entropy_end=$(awk -v s="${num_env_steps}" 'BEGIN{printf "%.0f", s}')
entropy_coef_horizons="0 ${entropy_mid} ${entropy_end}"
reward_shaping_horizon="${num_env_steps}"

# Eval envs are batched two per worker process by train_adaptive.py, so this
# must be even; 2x the population gives every partner two threads per pass.
n_eval_rollout_threads=$((population_size * 2))

# The runner logs metrics and evaluates on an *episode* (PPO update) counter, not
# a step counter, and one episode is episode_length * n_rollout_threads steps --
# so the upstream literals silently become coarser the more rollout threads you
# give it. At 16 threads a 1e6-step run is 156 updates, and `--log_interval 50`
# yields three points of training curve. Derive both from the budget instead:
# ~60 logged points and ~8 evals whatever the thread count.
episodes=$(awk -v s="${num_env_steps}" -v e=400 -v t="${ROLLOUT_THREADS}" 'BEGIN{printf "%d", s/(e*t)}')
log_interval=$(awk -v n="${episodes}" 'BEGIN{v=int(n/60); print (v<1)?1:v}')
eval_interval=$(awk -v n="${episodes}" 'BEGIN{v=int(n/8); print (v<1)?1:v}')
# eval_episodes is multiplied by the population size inside the runner, so 8
# here is 8*population episodes per pass. The upstream 32 would spend as much
# compute evaluating as training on a short run.
eval_episodes=8

num_agents=2
algo="adaptive"
exp="fcp-S2-${arm}${exp_suffix}"

# Partner-conditioning ablation. PID_OBS=1 shows the ego agent which partner it
# is paired with, on top of the critic-side --use_agent_policy_id every stage-2
# run already gets. This is an ORACLE UPPER BOUND, not a zero-shot method: a
# held-out partner has no id the agent was ever trained on. See CLAUDE.md.
#
# The width must equal the number of entries in the population yml, because
# policy_pool.load_population assigns id = (i + 1) / len(population_config) --
# counting the trainable agent, not just the partners. Defaulting to that count
# removes the one real footgun; a wrong width raises rather than aliasing two
# partners onto one index.
pid_flags=""
if [[ -n "${PID_OBS}" ]]; then
    pid_dim=${PID_OBS_DIM:-$(grep -c "^[a-zA-Z_][a-zA-Z_0-9]*:" "${yml}")}
    pid_flags="--use_agent_policy_id_obs --agent_policy_id_obs_dim ${pid_dim}"
fi

ulimit -n 65536 || ulimit -n 4096

echo "env ${env}, layout ${layout}, arm ${arm}, population ${population_size}, seeds ${seed_begin}..${seed_max}, steps ${num_env_steps}"
echo "${episodes} updates at ${ROLLOUT_THREADS} rollout threads: log every ${log_interval}, eval every ${eval_interval}"
if [[ -n "${pid_flags}" ]]; then
    echo "partner-id observation: ${pid_flags}"
fi
echo "population yml: ${yml}"
for seed in $(seq ${seed_begin} ${seed_max}); do
    echo "=== ${exp} seed ${seed} ==="
    python train/train_adaptive.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" \
        --layout_name ${layout} --num_agents ${num_agents} \
        --seed ${seed} --n_training_threads $TRAINING_THREADS --n_rollout_threads $ROLLOUT_THREADS \
        --episode_length 400 --num_env_steps ${num_env_steps} \
        --reward_shaping_horizon ${reward_shaping_horizon} --overcooked_version ${version} \
        --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
        --stage 2 --data_parallel \
        --morl_objectives default \
        --save_interval 1000 --log_interval ${log_interval} \
        --use_eval --eval_interval ${eval_interval} --n_eval_rollout_threads ${n_eval_rollout_threads} --eval_episodes ${eval_episodes} \
        --population_yaml_path "${yml}" \
        --population_size ${population_size} --adaptive_agent_name fcp_adaptive --use_agent_policy_id \
        ${pid_flags} \
        --use_proper_time_limits \
        --wandb_tags morl-s2 ${arm} \
        --wandb_name $WANDB_ENTITY || exit 1
done
